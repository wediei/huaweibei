# -*- coding: utf-8 -*-
"""
多模型集成提交脚本

加载多个训练的 checkpoint，对每个测试位置取复数平均，生成提交文件。

用法:
    python ensemble_submit.py \
      --checkpoints output_r2_s42/xxx/best.pth output_r2_s43/xxx/best.pth output_r2_s44/xxx/best.pth \
      --data_dir Round1_Map \
      --output ./submission_ensemble
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(data_dir, train_config, device):
    """重建单个模型 (v5 几何接地版本)"""
    from scene.round1_dataset import Round1Dataset
    from scene.map_encoder import load_map_pointcloud, PositionalEncoder
    from scene.gaussian_model import GaussianModel
    from scene.geometry_encoder import GeometricFeatureComputer
    from scene.geometry_decoder import GeometryGroundedDecoder
    from train_round1 import ChannelPredictionModel

    # 加载标准化参数
    ds = Round1Dataset(data_dir, split='train', normalize_pos=True)
    pos_mean, pos_std = ds.pos_mean, ds.pos_std

    # 地图点云
    map_points = load_map_pointcloud(data_dir)
    x_min, x_max = ds.positions[:, 0].min(), ds.positions[:, 0].max()
    y_min, y_max = ds.positions[:, 1].min(), ds.positions[:, 1].max()
    mask = (map_points[:, 0] >= x_min - 20) & (map_points[:, 0] <= x_max + 20) & \
           (map_points[:, 1] >= y_min - 20) & (map_points[:, 1] <= y_max + 20)
    map_points = map_points[mask]

    pe = PositionalEncoder(multires=10).to(device)

    # 高斯模型
    sh_deg = train_config.get('sh_degree', 1)
    scene_extent = np.max(ds.positions.max(0) - ds.positions.min(0))
    gaussians = GaussianModel(sh_degree=sh_deg)
    gaussians.create_from_map(map_points, n_init=train_config.get('n_gaussians', 15000),
                               spatial_lr_scale=scene_extent)
    gfeat_raw = gaussians.get_features
    sh_dim_total = gfeat_raw.view(gfeat_raw.shape[0], -1).shape[-1]

    # 几何特征计算器
    geo_computer = GeometricFeatureComputer(
        bs_position=(50.0, 0.0, 25.0),
    ).to(device)

    # 几何接地解码器
    scatter_dim = sh_dim_total + 4
    decoder = GeometryGroundedDecoder(
        scatter_dim=scatter_dim,
        hidden_dim=128,
        rank=train_config.get('rank', 16),
        angle_bins=256,
        delay_bins=192,
        ue_ants=4,
    ).to(device)

    # 形变模型
    from scene.deform_model import DeformModel
    geo_feat_dim = 8
    deform_model = DeformModel(
        is_blender=False, is_6dof=False,
        map_feat_dim=0,
        gaussian_feat_dim=sh_dim_total,
        geo_feat_dim=geo_feat_dim,
    )

    model = ChannelPredictionModel(
        gaussian_model=gaussians,
        channel_decoder=decoder,
        deform_model=deform_model.deform,
        pos_encoder=pe,
        geo_computer=geo_computer,
        sh_dim_total=sh_dim_total,
    ).to(device)

    return model, pos_mean, pos_std


@torch.no_grad()
def predict_single(model, test_loader, pos_mean, pos_std, device):
    """单模型预测"""
    model.eval()
    pos_mean_t = torch.from_numpy(pos_mean).float().to(device)
    pos_std_t = torch.from_numpy(pos_std).float().to(device)

    all_h = []
    for batch in tqdm(test_loader, desc='Predict', leave=False):
        pos = batch.to(device)

        # v5: 几何特征需要世界坐标
        pos_raw = pos * pos_std_t + pos_mean_t

        hr, hi = model(pos, pos_raw=pos_raw)
        h_complex = (hr + 1j * hi).cpu().numpy()
        all_h.append(h_complex)

    return np.concatenate(all_h, axis=0)  # (500, 256, 4, 192)


def main():
    parser = ArgumentParser(description='多模型集成提交')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='checkpoint 路径列表')
    parser.add_argument('--data_dir', type=str, default='Round1_Map')
    parser.add_argument('--output', type=str, default='./submission_ensemble')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='每模型权重 (默认等权)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 测试数据
    from scene.round1_dataset import Round1Dataset
    ds = Round1Dataset(args.data_dir, split='test', normalize_pos=True)
    test_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f'Test samples: {len(ds)}')

    weights = args.weights if args.weights else [1.0] * len(args.checkpoints)
    weights = np.array(weights) / np.sum(weights)

    all_h = []
    for i, ckpt_path in enumerate(args.checkpoints):
        print(f'\n{"="*50}')
        print(f'Model {i+1}/{len(args.checkpoints)}: {ckpt_path}')

        # 加载配置
        config_dir = os.path.dirname(os.path.dirname(ckpt_path))
        config_path = os.path.join(config_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                train_config = json.load(f)
        else:
            print(f'[WARN] No config at {config_path}, using defaults')
            train_config = {}

        # 重建模型
        print(f'  Building model...')
        model, pos_mean, pos_std = build_model(args.data_dir, train_config, device)

        # 加载权重
        print(f'  Loading weights...')
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        ep = checkpoint.get('epoch', '?')
        sc = checkpoint.get('best_score', '?')
        print(f'  Loaded epoch {ep}, score={sc}')

        # 推理
        print(f'  Running inference...')
        h_i = predict_single(model, test_loader, pos_mean, pos_std, device)
        all_h.append(h_i)
        del model
        torch.cuda.empty_cache()

    # 加权平均
    print(f'\n{"="*50}')
    print(f'Ensemble averaging ({len(all_h)} models, weights={weights})')
    h_ensemble = np.zeros_like(all_h[0], dtype=np.complex128)
    for h_i, w_i in zip(all_h, weights):
        h_ensemble += w_i * h_i
    h_ensemble = h_ensemble.astype(np.complex64)

    output_path = os.path.join(args.output, 'Round1_Test_Channel.npy')
    np.save(output_path, h_ensemble)
    print(f'  Saved: {output_path}')
    print(f'  Shape: {h_ensemble.shape}, Size: {os.path.getsize(output_path)/1e6:.2f} MB')

    # 保存集成信息
    info = {
        'checkpoints': args.checkpoints,
        'weights': weights.tolist(),
        'n_models': len(args.checkpoints),
    }
    with open(os.path.join(args.output, 'ensemble_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f'  Info saved.')


if __name__ == '__main__':
    main()
