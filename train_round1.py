# -*- coding: utf-8 -*-
"""
Round1 无线信道大赛 — 训练脚本

基于 WRF-GS+ 的 3DGS 框架，适配 MIMO-OFDM 信道预测任务。

用法:
    python train_round1.py --data_dir Round1_Map --output_dir ./output_round1

架构:
    - MapEncoder: 从 .ply 点云提取位置感知特征
    - GaussianModel: 可学习的 3D 场景表征
    - DeformModel: 查询位置 → per-gaussian 形变
    - GaussianFeatureAggregator: 距离加权聚合高斯特征
    - ChannelDecoder: 聚合特征 → MIMO-OFDM 信道矩阵
    - ChannelLoss: PAS + PDP + NMSE 多目标损失
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from argparse import ArgumentParser
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================== 完整模型 ====================

class ChannelPredictionModel(nn.Module):
    """
    完整信道预测模型 — v5 几何接地版本

    架构:
      1. PositionalEncoder: 查询位置 → 位置编码
      2. GeometricFeatureComputer: 高斯XYZ + 查询XYZ → 方向/距离/延迟 (0参数, 纯几何)
      3. DeformNetwork: (高斯XYZ, 查询位置, 几何特征) → per-gaussian 调制 (WRF-GS+ 资产)
      4. GeometryGroundedDecoder: 散射向量 → 角度/延迟软投影 → CP 重建 → H
    """

    def __init__(self, gaussian_model, channel_decoder, deform_model,
                 pos_encoder, geo_computer, sh_dim_total):
        super().__init__()
        self.gaussian_model = gaussian_model
        self.channel_decoder = channel_decoder    # GeometryGroundedDecoder
        self.deform_model = deform_model          # DeformNetwork
        self.pos_encoder = pos_encoder
        self.geo_computer = geo_computer          # GeometricFeatureComputer
        self.sh_dim_total = sh_dim_total          # 总 SH 维度 (sh_d=1→12, sh_d=2→27)

    def forward(self, query_pos, return_components=False, pos_raw=None):
        """
        Args:
            query_pos: (B, 3) 查询位置 (归一化后)
            return_components: 是否返回中间特征
            pos_raw: (B, 3) 世界坐标 (GeometricFeatureComputer 需要)

        Returns:
            (h_real, h_imag): 各 (B, 256, 4, 192)
        """
        B = query_pos.shape[0]
        N = self.gaussian_model.get_xyz.shape[0]

        # ---- 1. 几何特征 (地图利用!) ----
        xyz_world = self.gaussian_model.get_xyz.detach()  # (N, 3) — 被锚定在地图上
        geo_feats = self.geo_computer(xyz_world, pos_raw)  # dict of (B,N,*) tensors

        # ---- 2. DeformNetwork: 查询相关的高斯调制 (WRF-GS+ 资产) ----
        xyz_exp = xyz_world.unsqueeze(0).expand(B, -1, -1).reshape(-1, 3)  # (B*N, 3)
        pos_exp = query_pos.unsqueeze(1).expand(-1, N, -1).reshape(-1, 3)  # (B*N, 3)

        # 几何特征拼接给 DeformNetwork
        geo_input = torch.cat([
            geo_feats['dir_dep'].reshape(-1, 3),
            geo_feats['dir_arr'].reshape(-1, 3),
            geo_feats['log_d_bs'].reshape(-1, 1),
            geo_feats['log_d_ue'].reshape(-1, 1),
        ], dim=-1)  # (B*N, 8)

        d_xyz, d_rotation, d_scaling, d_signal = self.deform_model(
            xyz_exp, pos_exp, geo_feat=geo_input
        )

        # ---- 3. 构建散射向量 ----
        # SH 特征 + 调制
        sh_feat = self.gaussian_model.get_features  # (N, C_sh, 3)
        sh_flat = sh_feat.view(N, -1).unsqueeze(0).expand(B, -1, -1)  # (B, N, sh_dim)

        d_signal_r = d_signal.view(B, N, -1)  # (B, N, sh_dim)
        sh_modulated = sh_flat + 0.1 * d_signal_r  # (B, N, sh_dim)

        # 不透明度 (散射强度)
        opacity = self.gaussian_model.get_opacity  # (N, 1)
        opacity_b = (opacity.squeeze(-1).unsqueeze(0).unsqueeze(-1)
                     .expand(B, -1, 1))  # (B, N, 1)

        # 拼接: SH特征 + 不透明度 + 散射角 + 对数距离
        scatter_vec = torch.cat([
            sh_modulated,
            opacity_b,
            geo_feats['cos_scat'],
            geo_feats['log_d_bs'],
            geo_feats['log_d_ue'],
        ], dim=-1)  # (B, N, sh_dim + 1 + 1 + 1 + 1)

        # ---- 4. 几何接地解码 ----
        h_real, h_imag = self.channel_decoder(
            scatter_vec, geo_feats['az_dep'], geo_feats['tau'].squeeze(-1)
        )

        if return_components:
            return h_real, h_imag, {
                'geo_feats': {k: v for k, v in geo_feats.items()
                             if isinstance(v, torch.Tensor)},
            }

        return h_real, h_imag


# ==================== 训练函数 ====================

def train_epoch(model, loader, criterion, optimizer, device, epoch, args=None, pos_mean=None, pos_std=None):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_metrics = {'pas_cos': 0, 'pdp_cos': 0, 'nmse': 0, 'score': 0}
    n_batches = len(loader)
    log_interval = getattr(args, 'log_interval', 50) if args else 50

    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for batch_idx, batch in enumerate(pbar):
        pos, ch_real, ch_imag = batch
        pos = pos.to(device)

        # 数据增强: 对位置加噪声 (只在训练时)
        pos_aug = pos
        if args is not None and getattr(args, 'augment', False):
            noise = torch.randn_like(pos) * args.augment_std
            pos_aug = pos + noise

        # 原始坐标 (GeometricFeatureExtractor 和 LOSEncoder 需要世界坐标)
        pos_raw = None
        if pos_mean is not None and pos_std is not None:
            pos_raw = pos_aug * pos_std + pos_mean  # 反标准化到世界坐标

        ch_gt = torch.stack([ch_real, ch_imag], dim=1).to(device)

        optimizer.zero_grad()

        # 前向
        h_real, h_imag = model(pos_aug, pos_raw=pos_raw)
        h_pred = torch.stack([h_real, h_imag], dim=1)

        # 主损失
        loss, loss_dict = criterion(h_pred, ch_gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 密度控制: 复用 3DGS 框架, 根据 xyz 梯度自适应增减高斯
        # 仅在非 fix_xyz 模式 + 指定间隔触发
        if (args is not None and not getattr(args, 'fix_xyz', False) and
            getattr(args, 'densify_step', 0) > 0 and
            batch_idx == 0 and epoch > 0 and
            epoch % args.densify_step == 0):
            gauss_xyz = model.gaussian_model.get_xyz
            scene_ext = gauss_xyz.std().item() * 3
            # 用 opacity 梯度的 norm 作为重要性信号
            if model.gaussian_model._opacity.grad is not None:
                grad_signal = model.gaussian_model._opacity.grad.abs()
                xyz_grad = model.gaussian_model._xyz.grad
                if xyz_grad is not None:
                    # 累加梯度统计
                    acc = xyz_grad.norm(dim=-1, keepdim=True)
                    if not hasattr(model.gaussian_model, 'xyz_gradient_accum'):
                        model.gaussian_model.xyz_gradient_accum = torch.zeros_like(acc)
                        model.gaussian_model.denom = torch.zeros_like(acc)
                    model.gaussian_model.xyz_gradient_accum += acc
                    model.gaussian_model.denom += 1
                    grad_mean = model.gaussian_model.xyz_gradient_accum / model.gaussian_model.denom.clamp(1)
                    # 拆分高梯度大高斯
                    threshold = grad_mean.median() * 3
                    size_thresh = scene_ext * 0.01
                    mask_split = (grad_mean.squeeze() > threshold) & (model.gaussian_model.get_scaling.max(dim=1).values > size_thresh)
                    # 克隆高梯度小高斯
                    mask_clone = (grad_mean.squeeze() > threshold) & (model.gaussian_model.get_scaling.max(dim=1).values <= size_thresh)
                    # 删除低不透明度高斯
                    mask_prune = (model.gaussian_model.get_opacity.squeeze() < 0.005)
                    n_split = mask_split.sum().item()
                    n_clone = mask_clone.sum().item()
                    n_prune = mask_prune.sum().item()
                    if n_split + n_clone + n_prune > 0:
                        print(f'  [Densify e{epoch}] split={n_split} clone={n_clone} prune={n_prune} (total={gauss_xyz.shape[0]})')

        # 统计
        total_loss += loss.item()
        metrics = criterion.compute_metrics(h_pred, ch_gt)
        for k in total_metrics:
            total_metrics[k] += metrics[k]

        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pas': f'{metrics["pas_cos"]:.4f}',
                'pdp': f'{metrics["pdp_cos"]:.4f}',
                'nmse': f'{metrics["nmse"]:.6f}',
            })

    avg_loss = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, pos_mean=None, pos_std=None):
    """评估模型"""
    model.eval()
    total_metrics = {'pas_cos': 0, 'pdp_cos': 0, 'nmse': 0, 'score': 0}
    total_loss = 0
    n_batches = len(loader)

    for batch in tqdm(loader, desc='Evaluation', leave=False):
        pos, ch_real, ch_imag = batch
        pos = pos.to(device)
        ch_gt = torch.stack([ch_real, ch_imag], dim=1).to(device)

        # 计算世界坐标 (几何特征需要)
        pos_raw = None
        if pos_mean is not None:
            pos_raw = pos * pos_std + pos_mean

        h_real, h_imag = model(pos, pos_raw=pos_raw)
        h_pred = torch.stack([h_real, h_imag], dim=1)

        loss, _ = criterion(h_pred, ch_gt)
        total_loss += loss.item()

        metrics = criterion.compute_metrics(h_pred, ch_gt)
        for k in total_metrics:
            total_metrics[k] += metrics[k]

    n = max(n_batches, 1)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


# ==================== 主函数 ====================

def main():
    parser = ArgumentParser(description='Round1 信道预测训练')
    parser.add_argument('--data_dir', type=str, default='Round1_Map',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./output_round1',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--n_gaussians', type=int, default=15000,
                        help='初始高斯点数')
    parser.add_argument('--map_feat_dim', type=int, default=32,
                        help='地图特征维度')
    parser.add_argument('--n_map_ref', type=int, default=30000,
                        help='地图参考点数')
    parser.add_argument('--knn_k', type=int, default=16,
                        help='KNN 近邻数')
    parser.add_argument('--sh_degree', type=int, default=1,
                        help='球谐函数阶数 (0=3维特征, 1=12维特征)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='日志间隔')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='评估间隔 (epoch)')
    parser.add_argument('--no_val', action='store_true', default=False,
                        help='不使用验证集，用全部 2000 样本训练')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='保存间隔 (epoch)')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的 checkpoint 路径')
    # 数据增强
    parser.add_argument('--augment', action='store_true', default=False,
                        help='训练时对位置加噪声增强')
    parser.add_argument('--augment_std', type=float, default=0.5,
                        help='噪声标准差 (米)')
    # LOS 可视角编码
    parser.add_argument('--use_los', action='store_true', default=False,
                        help='使用体素射线可视角编码')
    # 几何特征编码
    parser.add_argument('--use_geo', action='store_true', default=False,
                        help='使用 GeometricFeatureExtractor 替代 MapPointFeature')
    parser.add_argument('--anchor_weight', type=float, default=0.01,
                        help='高斯锚定损失权重')
    parser.add_argument('--anchor_interval', type=int, default=5,
                        help='锚定损失计算间隔 (epoch)')
    parser.add_argument('--anchor_max_dist', type=float, default=2.0,
                        help='最大允许漂移距离 (米)')
    # 高斯锚定
    parser.add_argument('--fix_xyz', action='store_true', default=False,
                        help='固定高斯位置在地图几何上 (lr≈0)')
    parser.add_argument('--densify_step', type=int, default=500,
                        help='密度控制间隔 (0=禁用)')
    parser.add_argument('--rank', type=int, default=16,
                        help='ChannelDecoder 低秩分解秩数')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # 保存配置
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ========== 1. 数据加载 ==========
    print('\n=== Loading Data ===')
    from scene.round1_dataset import Round1Dataset

    train_dataset = Round1Dataset(args.data_dir, split='train', normalize_pos=True)
    # 保存原始位置数据（random_split 后 Subset 不保留 .positions）
    all_positions = train_dataset.positions.copy()

    if args.no_val:
        # 全样本训练：不使用验证集
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = None
        best_score = -1.0
    else:
        # 从训练集划分验证集 (5%) — 测试集没有信道标签无法评估
        n_train = len(train_dataset)
        n_val = max(1, int(n_train * 0.05))
        n_train_new = n_train - n_val
        train_subset, val_dataset = torch.utils.data.random_split(
            train_dataset, [n_train_new, n_val],
            generator=torch.Generator().manual_seed(args.seed)
        )
        # 继承标准化参数
        val_dataset.dataset.pos_mean = train_dataset.pos_mean
        val_dataset.dataset.pos_std = train_dataset.pos_std

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        best_score = -1.0

    # 测试集加载器 (仅位置，用于最终提交)
    test_dataset = Round1Dataset(args.data_dir, split='test', normalize_pos=True,
                                  pos_mean=train_dataset.pos_mean,
                                  pos_std=train_dataset.pos_std)

    pos_mean = torch.from_numpy(train_dataset.pos_mean).float().to(device)
    pos_std = torch.from_numpy(train_dataset.pos_std).float().to(device)

    # ========== 2. 地图加载 + 位置编码 ==========
    print('\n=== Loading Map + PositionalEncoder ===')
    from scene.map_encoder import load_map_pointcloud, PositionalEncoder

    map_points = load_map_pointcloud(args.data_dir)
    # 使用 all_positions 裁剪地图（原始完整数据集的范围）
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    margin = 20  # 扩展20m
    mask = (map_points[:, 0] >= x_min - margin) & (map_points[:, 0] <= x_max + margin) & \
           (map_points[:, 1] >= y_min - margin) & (map_points[:, 1] <= y_max + margin)
    map_points = map_points[mask]
    print(f'  Cropped map points: {map_points.shape[0]}')

    pos_encoder = PositionalEncoder(multires=10).to(device)

    # ========== 3. 高斯模型初始化 (锚定到地图) ==========
    print('\n=== Initializing Gaussians (Map-Anchored) ===')
    from scene.gaussian_model import GaussianModel

    scene_extent = np.max(all_positions.max(axis=0) - all_positions.min(axis=0))
    gaussians = GaussianModel(sh_degree=args.sh_degree, optimizer_type='default')
    gaussians.create_from_map(map_points, n_init=args.n_gaussians,
                               spatial_lr_scale=scene_extent,
                               fix_xyz=args.fix_xyz,
                               sh_degree_override=args.sh_degree)
    gfeat_raw = gaussians.get_features
    sh_dim_total = gfeat_raw.view(gfeat_raw.shape[0], -1).shape[-1]  # sh_d=1→12, sh_d=2→27
    print(f'  Gaussians: {gaussians.get_xyz.shape[0]} points, SH dim={sh_dim_total}')

    # ========== 4. 几何特征计算器 (0 参数, 纯地图利用) ==========
    print('\n=== Building GeometricFeatureComputer ===')
    from scene.geometry_encoder import GeometricFeatureComputer
    geo_computer = GeometricFeatureComputer(
        bs_position=(50.0, 0.0, 25.0),
    ).to(device)

    # ========== 5. 形变模型 (含几何特征输入) ==========
    print('\n=== Building Deform Model (with geometry input) ===')
    from scene.deform_model import DeformModel

    geo_feat_dim = 8  # dir_dep(3) + dir_arr(3) + log_d_bs(1) + log_d_ue(1)
    deform_model = DeformModel(is_blender=False, is_6dof=False,
                                map_feat_dim=0,          # 不用 KNN map feat, 用几何特征替代
                                gaussian_feat_dim=sh_dim_total,
                                geo_feat_dim=geo_feat_dim)

    # ========== 6. 几何接地解码器 (替代 MLP ChannelDecoder) ==========
    print('\n=== Building GeometryGroundedDecoder ===')
    from scene.geometry_decoder import GeometryGroundedDecoder

    scatter_dim = sh_dim_total + 4  # SH + opacity + cos_scat + log_d_bs + log_d_ue
    decoder = GeometryGroundedDecoder(
        scatter_dim=scatter_dim,
        hidden_dim=128,
        rank=args.rank,
        angle_bins=256,
        delay_bins=192,
        ue_ants=4,
    ).to(device)

    # ========== 7. 完整模型 ==========
    print('\n=== Assembling Full Model ===')
    model = ChannelPredictionModel(
        gaussian_model=gaussians,
        channel_decoder=decoder,
        deform_model=deform_model.deform,
        pos_encoder=pos_encoder,
        geo_computer=geo_computer,
        sh_dim_total=sh_dim_total,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  Total params: {total_params:,}')
    print(f'  Trainable params: {trainable_params:,}')

    # ========== 7. 优化器和损失 ==========
    from utils.channel_loss import ChannelLoss

    # 参数分组 (高斯参数独立优化)
    param_groups = [
        {'params': model.gaussian_model._xyz, 'lr': args.lr * (0.001 if args.fix_xyz else 0.5), 'name': 'xyz'},
        {'params': model.gaussian_model._features_dc, 'lr': args.lr, 'name': 'feat_dc'},
        {'params': model.gaussian_model._features_rest, 'lr': args.lr * 0.05, 'name': 'feat_rest'},
        {'params': model.gaussian_model._opacity, 'lr': args.lr * 0.1, 'name': 'opacity'},
        {'params': model.gaussian_model._scaling, 'lr': args.lr * 0.1, 'name': 'scaling'},
        {'params': model.gaussian_model._rotation, 'lr': args.lr * 0.1, 'name': 'rotation'},
        {'params': model.channel_decoder.parameters(), 'lr': args.lr, 'name': 'decoder'},
        {'params': model.deform_model.parameters(), 'lr': args.lr * 0.5, 'name': 'deform'},
    ]

    optimizer = torch.optim.Adam(param_groups, eps=1e-8)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    criterion = ChannelLoss(w_pas=0.4, w_pdp=0.4, w_nmse=0.2, use_real_imag=True, nmse_clip=20.0).to(device)

    # ========== 8. 恢复 checkpoint ==========
    start_epoch = 0
    best_score = -1.0
    if args.resume:
        print(f'\n=== Resuming from {args.resume} ===')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', -1.0)
        print(f'  Resumed at epoch {start_epoch}, best_score={best_score:.4f}')

    # ========== 9. 训练循环 ==========
    print('\n=== Training ===')
    log_file = os.path.join(output_dir, 'training_log.txt')

    for epoch in range(start_epoch, args.epochs):
        t_start = time.time()

        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            args=args, pos_mean=pos_mean, pos_std=pos_std,
        )

        epoch_time = time.time() - t_start

        # 日志
        log_msg = (f'Epoch {epoch:3d}/{args.epochs} | '
                   f'Time: {epoch_time:.1f}s | '
                   f'Train Loss: {train_loss:.4f} | '
                   f'PAS: {train_metrics["pas_cos"]:.4f} | '
                   f'PDP: {train_metrics["pdp_cos"]:.4f} | '
                   f'NMSE: {train_metrics["nmse"]:.6f}')
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')

        scheduler_cosine.step()

        # 输出缩放因子日志
        if epoch == 0 or (epoch + 1) % 10 == 0:
            scale_val = torch.exp(model.channel_decoder.output_log_scale).item()
            print(f'  Output scale: {scale_val:.6f}')

        # 评估
        if val_loader is not None and ((epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1):
            eval_loss, eval_metrics = evaluate(model, val_loader, criterion, device,
                                                   pos_mean=pos_mean, pos_std=pos_std)

            score = eval_metrics['score']
            eval_msg = (f'  Eval: Loss: {eval_loss:.4f} | '
                        f'PAS: {eval_metrics["pas_cos"]:.4f} | '
                        f'PDP: {eval_metrics["pdp_cos"]:.4f} | '
                        f'NMSE: {eval_metrics["nmse"]:.6f} | '
                        f'Score: {score:.4f}')
            print(f'  {"=" * 50}')
            print(eval_msg)
            print(f'  {"=" * 50}')
            with open(log_file, 'a') as f:
                f.write(eval_msg + '\n')

            # 保存最佳模型
            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score,
                    'eval_metrics': eval_metrics,
                    'args': vars(args),
                }, os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
                print(f'  ** New best model saved! Score: {best_score:.4f}')

        # 定期保存 (可选)
        if (epoch + 1) % args.save_interval == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
            }, os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth'))

    print(f'\n=== Training Complete ===')
    print(f'Best score: {best_score:.4f}')
    print(f'Output: {output_dir}')
    # --no_val 模式：保存最终模型
    if args.no_val:
        final_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
        torch.save({
            'epoch': args.epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': -1.0,
            'args': vars(args),
        }, final_path)
        print(f'Final model saved to: {final_path}')
    print(f'Best model: {os.path.join(output_dir, "checkpoints", "best_model.pth")}')


if __name__ == '__main__':
    main()
