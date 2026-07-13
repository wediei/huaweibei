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
    完整信道预测模型

    将 MapEncoder + 高斯表征 + DeformNetwork + 聚合 + Decoder 整合
    """

    def __init__(self, gaussian_model, map_encoder, channel_decoder, deform_model,
                 pos_encoder, gaussian_aggregator):
        super().__init__()
        self.gaussian_model = gaussian_model
        self.map_encoder = map_encoder
        self.channel_decoder = channel_decoder
        self.deform_model = deform_model
        self.pos_encoder = pos_encoder
        self.gaussian_aggregator = gaussian_aggregator

    def forward(self, query_pos, return_components=False):
        """
        Args:
            query_pos: (B, 3) 查询位置 (归一化后)
            return_components: 是否返回中间特征（用于调试）

        Returns:
            h_pred: (B, 256, 4, 192) complex64 或
            (h_real, h_imag): (B, 256, 4, 192) float32 × 2
        """
        B = query_pos.shape[0]
        N = self.gaussian_model.get_xyz.shape[0]

        # 1) 位置编码 + 地图特征
        pos_enc = self.pos_encoder(query_pos)  # (B, 63)

        # 地图特征: 每个查询位置从 .ply 参考点聚合
        map_feat = self.map_encoder(query_pos)  # (B, feat_dim)

        # 2) 高斯形变: 查询位置 → per-gaussian 偏移
        #    扩展 query_pos 到每个高斯: (B, 3) -> (B, N, 3) -> (B*N, 3)
        xyz = self.gaussian_model.get_xyz.detach()  # (N, 3)
        time_input = query_pos.unsqueeze(1).expand(-1, N, -1).reshape(-1, 3)  # (B*N, 3)
        xyz_expand = xyz.unsqueeze(0).expand(B, -1, -1).reshape(-1, 3)  # (B*N, 3)

        # 地图特征也扩展到每个高斯
        map_feat_expand = map_feat.unsqueeze(1).expand(-1, N, -1).reshape(-1, map_feat.shape[-1])

        # 注意: self.deform_model 是 DeformNetwork (不是 DeformModel 包装器)
        # 所以直接调用 forward(), 而非 .step()
        d_xyz, d_rotation, d_scaling, d_signal = self.deform_model(
            xyz_expand, time_input, map_feat=map_feat_expand
        )

        # 3) 应用形变到高斯
        xyz_deformed = xyz.unsqueeze(0) + d_xyz.view(B, N, 3)  # (B, N, 3)
        scaling = self.gaussian_model.get_scaling  # (N, 3)
        opacity = self.gaussian_model.get_opacity  # (N, 1)

        # 高斯特征: 在 SH degree=0 时为 (N, 1, 3), 展平为 (N, 3)
        feat = self.gaussian_model.get_features
        feat = feat.view(feat.shape[0], -1)  # (N, 3)

        # 信号调制: 将 d_signal 加到高斯特征上
        # d_signal: (B*N, 3) -> (B, N, 3)
        d_signal_reshaped = d_signal.view(B, N, -1)
        feat_modulated = feat.unsqueeze(0) + d_signal_reshaped  # (B, N, 3)

        # 4) 特征聚合: 对 batch 内每个位置分别加权聚合高斯特征
        # 使用 vmap 风格处理，避免 for 循环 (但对小 batch 直接 loop 更稳定)
        agg_feat_list = []
        last_weights = None
        for b in range(B):
            feat_b, weights_b = self.gaussian_aggregator(
                query_pos[b:b+1],  # (1, 3)
                xyz_deformed[b],   # (N, 3)
                opacity,
                scaling,
                feat_modulated[b],  # (N, 3)
            )
            agg_feat_list.append(feat_b)
            last_weights = weights_b
        agg_feat = torch.cat(agg_feat_list, dim=0)  # (B, 3)

        # 5) 全连接特征: 位置编码 + 地图特征 + 聚合特征
        decoder_input = torch.cat([pos_enc, map_feat, agg_feat], dim=-1)  # (B, 63+feat_dim+3)

        # 6) 信道解码
        h_real, h_imag = self.channel_decoder(decoder_input)  # 各 (B, 256, 4, 192)

        if return_components:
            return h_real, h_imag, {
                'pos_enc': pos_enc,
                'map_feat': map_feat,
                'agg_feat': agg_feat,
                'weights': last_weights if last_weights is not None else torch.zeros(1),
            }

        return h_real, h_imag


# ==================== 训练函数 ====================

def train_epoch(model, loader, criterion, optimizer, device, epoch, log_interval=50):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_metrics = {'pas_cos': 0, 'pdp_cos': 0, 'nmse': 0, 'score': 0}
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for batch_idx, batch in enumerate(pbar):
        pos, ch_real, ch_imag = batch
        pos = pos.to(device)
        ch_gt = torch.stack([ch_real, ch_imag], dim=1).to(device)  # (B, 2, 256, 4, 192)

        optimizer.zero_grad()

        # 前向
        h_real, h_imag = model(pos)
        h_pred = torch.stack([h_real, h_imag], dim=1)  # (B, 2, 256, 4, 192)

        # 损失
        loss, loss_dict = criterion(h_pred, ch_gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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
def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_metrics = {'pas_cos': 0, 'pdp_cos': 0, 'nmse': 0, 'score': 0}
    total_loss = 0
    n_batches = len(loader)

    for batch in tqdm(loader, desc='Evaluation', leave=False):
        pos, ch_real, ch_imag = batch
        pos = pos.to(device)
        ch_gt = torch.stack([ch_real, ch_imag], dim=1).to(device)

        h_real, h_imag = model(pos)
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
    parser.add_argument('--sh_degree', type=int, default=0,
                        help='球谐函数阶数 (0=无方向性)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='日志间隔')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='评估间隔 (epoch)')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='保存间隔 (epoch)')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的 checkpoint 路径')
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
    test_dataset = Round1Dataset(args.data_dir, split='test', normalize_pos=True,
                                  pos_mean=train_dataset.pos_mean,
                                  pos_std=train_dataset.pos_std)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    pos_mean = torch.from_numpy(train_dataset.pos_mean).float().to(device)
    pos_std = torch.from_numpy(train_dataset.pos_std).float().to(device)

    # ========== 2. 地图编码器 ==========
    print('\n=== Building Map Encoder ===')
    from scene.map_encoder import load_map_pointcloud, MapPointFeature, PositionalEncoder

    map_points = load_map_pointcloud(args.data_dir)
    # 只保留查询区域内的点 (加速)
    x_min, x_max = train_dataset.positions[:, 0].min(), train_dataset.positions[:, 0].max()
    y_min, y_max = train_dataset.positions[:, 1].min(), train_dataset.positions[:, 1].max()
    margin = 20  # 扩展20m
    mask = (map_points[:, 0] >= x_min - margin) & (map_points[:, 0] <= x_max + margin) & \
           (map_points[:, 1] >= y_min - margin) & (map_points[:, 1] <= y_max + margin)
    map_points = map_points[mask]
    print(f'  Cropped map points: {map_points.shape[0]}')

    map_encoder = MapPointFeature(map_points, n_ref_points=args.n_map_ref,
                                   feature_dim=args.map_feat_dim, knn_k=args.knn_k).to(device)
    pos_encoder = PositionalEncoder(multires=10).to(device)

    # ========== 3. 高斯模型初始化 ==========
    print('\n=== Initializing Gaussians ===')
    from scene.gaussian_model import GaussianModel

    scene_extent = np.max(train_dataset.positions.max(axis=0) - train_dataset.positions.min(axis=0))
    gaussians = GaussianModel(sh_degree=args.sh_degree, optimizer_type='default')
    # 使用地图点初始化高斯
    gaussians.create_from_map(map_points, n_init=args.n_gaussians,
                               spatial_lr_scale=scene_extent)
    print(f'  Gaussians: {gaussians.get_xyz.shape[0]} points')
    print(f'  Feature dim: {gaussians.get_features.shape[-1]}')

    # ========== 4. 形变模型 ==========
    print('\n=== Building Deform Model ===')
    from scene.deform_model import DeformModel

    gaussian_feat_dim = gaussians.get_features.shape[-1]  # 通常是3 (SH DC)
    deform_model = DeformModel(is_blender=False, is_6dof=False,
                                map_feat_dim=args.map_feat_dim,
                                gaussian_feat_dim=gaussian_feat_dim)

    # ========== 5. 聚合器和解码器 ==========
    print('\n=== Building Channel Decoder ===')
    from scene.channel_decoder import GaussianFeatureAggregator, ChannelDecoder

    aggregator = GaussianFeatureAggregator()

    # 解码器输入维度 = pos_enc(63) + map_feat + gaussian_agg_feat
    decoder_input_dim = pos_encoder.out_dim + args.map_feat_dim + gaussian_feat_dim
    decoder = ChannelDecoder(
        input_dim=decoder_input_dim,
        hidden_dims=[1024, 512, 512, 256],
        output_shape=(256, 4, 192)
    ).to(device)

    # ========== 6. 完整模型 ==========
    model = ChannelPredictionModel(
        gaussian_model=gaussians,
        map_encoder=map_encoder,
        channel_decoder=decoder,
        deform_model=deform_model.deform,  # 使用内部的 DeformNetwork
        pos_encoder=pos_encoder,
        gaussian_aggregator=aggregator,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  Total params: {total_params:,}')
    print(f'  Trainable params: {trainable_params:,}')

    # ========== 7. 优化器和损失 ==========
    from utils.channel_loss import ChannelLoss

    # 参数分组 (高斯参数独立优化)
    param_groups = [
        {'params': model.gaussian_model._xyz, 'lr': args.lr * 0.5, 'name': 'xyz'},
        {'params': model.gaussian_model._features_dc, 'lr': args.lr, 'name': 'feat_dc'},
        {'params': model.gaussian_model._opacity, 'lr': args.lr * 0.1, 'name': 'opacity'},
        {'params': model.gaussian_model._scaling, 'lr': args.lr * 0.1, 'name': 'scaling'},
        {'params': model.gaussian_model._rotation, 'lr': args.lr * 0.1, 'name': 'rotation'},
        {'params': model.map_encoder.parameters(), 'lr': args.lr, 'name': 'map_enc'},
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
            model, train_loader, criterion, optimizer, device, epoch, args.log_interval
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
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            eval_loss, eval_metrics = evaluate(model, test_loader, criterion, device)

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
    print(f'Best model: {os.path.join(output_dir, "checkpoints", "best_model.pth")}')


if __name__ == '__main__':
    main()
