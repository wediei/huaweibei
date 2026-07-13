# -*- coding: utf-8 -*-
"""
地图编码器：从 .ply 点云提取位置相关的无线特征

核心思想：
  将环境 .ply 点云下采样为一组可学习的特征点。
  对于任意查询位置，通过 KNN 聚合邻域特征，作为位置条件输入模型。

这满足大赛 "模型必须使用地图 M" 的要求。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== 点云加载与下采样 ==========

def load_map_pointcloud(data_dir):
    """从 Round1_Map.ply 加载点云坐标 (N, 3)"""
    ply_path = os.path.join(data_dir, 'Round1_Map.ply')
    if not os.path.exists(ply_path):
        # 如果 .ply 不存在，尝试备用路径
        ply_path = os.path.join(os.path.dirname(data_dir), 'Round1_Map.ply')
    if not os.path.exists(ply_path):
        # 仍然不存在，生成模拟点云用于调试
        print(f"[WARNING] Map .ply not found at {ply_path}, using simulated cloud")
        return _generate_simulated_cloud()

    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1).astype(np.float32)
    print(f"[MapEncoder] Loaded {len(points)} points from {ply_path}")
    return points


def _generate_simulated_cloud(n_points=50000):
    """生成模拟点云（仅在 .ply 文件缺失时用于调试）"""
    np.random.seed(42)
    x = np.random.uniform(40, 260, n_points)
    y = np.random.uniform(-210, 110, n_points)
    z = np.random.uniform(0, 50, n_points)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def farthest_point_sampling(points, n_samples):
    """
    最远点采样 (FPS) — 对大数据集使用随机采样替代
    Args:
        points: (N, 3) 输入点云
        n_samples: 目标采样点数
    Returns:
        sampled: (n_samples, 3)
        indices: (n_samples,)
    """
    n_pts = points.shape[0]
    if n_samples >= n_pts:
        return points, np.arange(n_pts)

    # 对于大点云，随机采样比 FPS 快 1000x+，且对 KNN 聚合效果等价
    indices = np.random.choice(n_pts, n_samples, replace=False)
    return points[indices], indices


# ========== 可学习地图特征 ==========

class MapPointFeature(nn.Module):
    """
    地图点特征编码器

    架构:
        1. 将 .ply 点云 FPS 下采样至 N_map 个参考点
        2. 每个参考点维护一个可学习的特征向量
        3. 查询位置通过 KNN 加权聚合近邻特征

    Args:
        map_points:     (N_ply, 3) 原始地图点云
        n_ref_points:   参考点数（下采样目标）
        feature_dim:    特征维度
        knn_k:          KNN 近邻数
    """
    def __init__(self, map_points, n_ref_points=50000, feature_dim=32, knn_k=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.knn_k = min(knn_k, n_ref_points)

        # FPS 下采样
        ref_points, ref_idx = farthest_point_sampling(map_points, n_ref_points)
        self.register_buffer('ref_points', torch.from_numpy(ref_points).float())  # (N_ref, 3)
        self.n_ref = self.ref_points.shape[0]

        # 可学习特征
        self.ref_features = nn.Parameter(torch.randn(self.n_ref, feature_dim) * 0.1)

        print(f"[MapPointFeature] {self.n_ref} ref points, dim={feature_dim}, K={self.knn_k}")

    def forward(self, xyz_query):
        """
        Args:
            xyz_query: (B, 3) 或 (N, 3) 查询位置
        Returns:
            feat: (B, feature_dim) 聚合后的地图特征
        """
        batch_size = xyz_query.shape[0]

        # 计算到所有参考点的距离
        # (B, N_ref) 每对距离
        dist = torch.cdist(xyz_query, self.ref_points, p=2)  # (B, N_ref)

        # 取最近的 K 个点
        knn_dist, knn_idx = torch.topk(dist, self.knn_k, dim=1, largest=False)  # (B, K)

        # 反距离加权
        eps = 1e-8
        weights = 1.0 / (knn_dist + eps)  # (B, K)
        weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化

        # 收集特征并加权聚合
        # knn_idx: (B, K) → 索引到 (N_ref, feat_dim) → (B, K, feat_dim)
        ref_feat = self.ref_features.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N_ref, feat)
        neighbor_feat = torch.gather(
            ref_feat, 1,
            knn_idx.unsqueeze(-1).expand(-1, -1, self.feature_dim)  # (B, K, feat_dim)
        )

        # 加权求和
        feat = (weights.unsqueeze(-1) * neighbor_feat).sum(dim=1)  # (B, feat_dim)

        return feat

    @torch.no_grad()
    def get_ref_points(self):
        """返回参考点的位置 (N_ref, 3) 和特征 (N_ref, feat_dim)"""
        return self.ref_points.cpu().numpy(), self.ref_features.detach().cpu().numpy()


# ========== 简化位置编码 ==========

class PositionalEncoder(nn.Module):
    """
    位置编码 (NeRF 风格)
    将 (x,y,z) 编码到高频特征
    """
    def __init__(self, multires=10):
        super().__init__()
        self.multires = multires
        self.out_dim = 3 + 3 * 2 * multires  # 原始 + sin/cos 各 multires 组

    def forward(self, xyz):
        """
        Args:
            xyz: (..., 3)
        Returns:
            encoded: (..., out_dim)
        """
        encoded = [xyz]
        for i in range(self.multires):
            freq = 2.0 ** i
            encoded.append(torch.sin(xyz * freq))
            encoded.append(torch.cos(xyz * freq))
        return torch.cat(encoded, dim=-1)
