# -*- coding: utf-8 -*-
"""
Round1 评估与提交脚本

用法:
    # 评估训练好的模型
    python evaluate.py --checkpoint output_round1/xxx/checkpoints/best_model.pth
    --data_dir Round1_Map --output ./submission

输出:
    Round1_Test_Channel.npy: (500, 256, 4, 192) complex64
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def predict_test_set(model, test_loader, device, batch_size=8, pos_mean=None, pos_std=None):
    """对测试集进行预测"""
    model.eval()

    pos_mean_t = torch.from_numpy(pos_mean).float().to(device) if pos_mean is not None else None
    pos_std_t = torch.from_numpy(pos_std).float().to(device) if pos_std is not None else None

    all_h_real = []
    all_h_imag = []

    for batch in tqdm(test_loader, desc='Predicting'):
        pos = batch.to(device)  # (B, 3)

        # 原始坐标 (LOS 需要)
        pos_raw = None
        if model.los_encoder is not None and pos_mean_t is not None:
            pos_raw = pos * pos_std_t + pos_mean_t

        h_real, h_imag = model(pos, pos_raw=pos_raw)  # 各 (B, 256, 4, 192)

        all_h_real.append(h_real.cpu().numpy())
        all_h_imag.append(h_imag.cpu().numpy())

    h_real = np.concatenate(all_h_real, axis=0)  # (500, 256, 4, 192)
    h_imag = np.concatenate(all_h_imag, axis=0)  # (500, 256, 4, 192)

    # 合并为复数
    h_complex = h_real.astype(np.float32) + 1j * h_imag.astype(np.float32)
    return h_complex.astype(np.complex64)


def main():
    parser = ArgumentParser(description='Round1 评估与提交')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--data_dir', type=str, default='Round1_Map',
                        help='数据目录')
    parser.add_argument('--output', type=str, default='./submission',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--config', type=str, default=None,
                        help='训练配置 JSON (可选，自动从输出目录读取)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # ========== 1. 加载配置 ==========
    # 尝试自动从 checkpoint 所在目录加载 config
    config_path = args.config
    if config_path is None:
        config_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        config_path = os.path.join(config_dir, 'config.json')

    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            train_config = json.load(f)
        print(f'Loaded config from {config_path}')
    else:
        print(f'[WARNING] Config not found at {config_path}, using defaults')
        train_config = {
            'sh_degree': 0,
            'map_feat_dim': 32,
            'n_map_ref': 30000,
            'knn_k': 16,
            'n_gaussians': 15000,
        }

    # ========== 2. 加载测试数据 ==========
    print('\n=== Loading Test Data ===')
    from scene.round1_dataset import Round1Dataset

    # 需要先加载训练集获取标准化参数
    train_dataset = Round1Dataset(args.data_dir, split='train', normalize_pos=True)
    test_dataset = Round1Dataset(args.data_dir, split='test', normalize_pos=True,
                                  pos_mean=train_dataset.pos_mean,
                                  pos_std=train_dataset.pos_std)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f'Test samples: {len(test_dataset)}')

    # ========== 3. 构建模型 ==========
    print('\n=== Building Model ===')
    from scene.map_encoder import load_map_pointcloud, PositionalEncoder
    from scene.gaussian_model import GaussianModel
    from scene.channel_decoder import GaussianFeatureAggregator, ChannelDecoder
    from train_round1 import ChannelPredictionModel

    map_points = load_map_pointcloud(args.data_dir)
    x_min, x_max = train_dataset.positions[:, 0].min(), train_dataset.positions[:, 0].max()
    y_min, y_max = train_dataset.positions[:, 1].min(), train_dataset.positions[:, 1].max()
    mask = (map_points[:, 0] >= x_min - 20) & (map_points[:, 0] <= x_max + 20) & \
           (map_points[:, 1] >= y_min - 20) & (map_points[:, 1] <= y_max + 20)
    map_points = map_points[mask]

    pos_encoder = PositionalEncoder(multires=10).to(device)

    # 高斯模型 (从配置文件恢复参数)
    sh_deg = train_config.get('sh_degree', 1)
    scene_extent = np.max(train_dataset.positions.max(axis=0) - train_dataset.positions.min(axis=0))
    gaussians = GaussianModel(sh_degree=sh_deg)
    gaussians.create_from_map(map_points, n_init=train_config.get('n_gaussians', 15000),
                               spatial_lr_scale=scene_extent,
                               sh_degree_override=sh_deg)

    # 解码器
    gaussian_feat_dim = gaussians.get_features.shape[-1]
    decoder_input_dim = pos_encoder.out_dim + train_config.get('map_feat_dim', 32) + gaussian_feat_dim

    decoder = ChannelDecoder(
        input_dim=decoder_input_dim,
        hidden_dims=[1024, 512, 512, 256],
        output_shape=(256, 4, 192)
    ).to(device)

    aggregator = GaussianFeatureAggregator()

    from scene.deform_model import DeformModel
    deform_model = DeformModel(
        is_blender=False, is_6dof=False,
        map_feat_dim=train_config.get('map_feat_dim', 32),
        gaussian_feat_dim=gaussian_feat_dim,
    )

    # map_encoder=None: 地图特征从高斯体 KNN 派生
    model = ChannelPredictionModel(
        gaussian_model=gaussians,
        map_encoder=None,
        channel_decoder=decoder,
        deform_model=deform_model.deform,
        pos_encoder=pos_encoder,
        gaussian_aggregator=aggregator,
    ).to(device)

    # ========== 4. 加载权重 ==========
    print(f'\n=== Loading Checkpoint: {args.checkpoint} ===')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    loaded_epoch = checkpoint.get("epoch", "?")
    loaded_score = checkpoint.get("best_score", None)
    score_str = f'{loaded_score:.4f}' if loaded_score is not None else 'N/A'
    print(f'  Loaded epoch {loaded_epoch}, score={score_str}')

    # ========== 5. 推理 ==========
    print('\n=== Running Inference ===')
    h_pred = predict_test_set(model, test_loader, device, args.batch_size,
                                pos_mean=train_dataset.pos_mean, pos_std=train_dataset.pos_std)
    print(f'  Prediction shape: {h_pred.shape}, dtype={h_pred.dtype}')

    # ========== 6. 保存提交 ==========
    output_path = os.path.join(args.output, 'Round1_Test_Channel.npy')
    np.save(output_path, h_pred)
    print(f'\n  Submitted file saved to: {output_path}')
    print(f'  File size: {os.path.getsize(output_path) / 1e6:.2f} MB')

    # ========== 7. 可选：在训练集上验证指标 ==========
    print('\n=== Validation on Train Set ===')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    from utils.channel_loss import ChannelLoss

    criterion = ChannelLoss(w_pas=0.4, w_pdp=0.4, w_nmse=0.2, use_real_imag=True).to(device)
    pos_mean_t = torch.from_numpy(train_dataset.pos_mean).float().to(device)
    pos_std_t = torch.from_numpy(train_dataset.pos_std).float().to(device)
    model.eval()

    all_metrics = {'pas_cos': 0, 'pdp_cos': 0, 'nmse': 0, 'score': 0}
    n_batches = 0

    for batch in tqdm(train_loader, desc='Validating'):
        pos, ch_real, ch_imag = batch
        pos = pos.to(device)
        ch_gt = torch.stack([ch_real, ch_imag], dim=1).to(device)

        pos_raw = None
        if model.los_encoder is not None:
            pos_raw = pos * pos_std_t + pos_mean_t
        h_real, h_imag = model(pos, pos_raw=pos_raw)
        h_pred = torch.stack([h_real, h_imag], dim=1)

        metrics = criterion.compute_metrics(h_pred, ch_gt)
        for k in all_metrics:
            all_metrics[k] += metrics[k]
        n_batches += 1

    print(f'\n  Validation Results:')
    for k, v in all_metrics.items():
        avg = v / max(n_batches, 1)
        print(f'    {k}: {avg:.6f}')

    # 保存评估结果
    result_path = os.path.join(args.output, 'validation_results.json')
    import json
    with open(result_path, 'w') as f:
        json.dump({k: v / max(n_batches, 1) for k, v in all_metrics.items()}, f, indent=2)
    print(f'  Results saved to: {result_path}')


if __name__ == '__main__':
    main()
