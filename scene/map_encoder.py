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


# ========== 射线可视角编码 (LOS) ==========

class LOSEncoder(nn.Module):
    """
    射线可视角编码器 — 真正利用地图几何信息

    原理:
        1. 将 .ply 点云体素化为占据网格 (1m 分辨率)
        2. 从基站 [50,0,25] 向查询位置发射射线
        3. 沿射线步进检测是否有体素被占据
        4. 输出: LOS 标志 (0/1) + 最近遮挡物距离

    Args:
        map_points:     (N_ply, 3) 原始地图点云
        bs_position:    基站位置 [50, 0, 25]
        voxel_size:     体素大小 (米)
        step_factor:    步进因子 (0.5 = 体素一半)
    """
    def __init__(self, map_points, bs_position=None, voxel_size=1.0, step_factor=0.5):
        super().__init__()
        if bs_position is None:
            bs_position = np.array([50.0, 0.0, 25.0], dtype=np.float32)
        self.register_buffer('bs_position', torch.from_numpy(np.array(bs_position, dtype=np.float32)))
        self.voxel_size = voxel_size
        self.step_size = voxel_size * step_factor

        # 网格边界
        x_min, x_max = map_points[:, 0].min(), map_points[:, 0].max()
        y_min, y_max = map_points[:, 1].min(), map_points[:, 1].max()
        z_min, z_max = 0.0, 50.0
        margin = 5.0
        grid_min = np.array([x_min - margin, y_min - margin, z_min], dtype=np.float32)
        grid_max = np.array([x_max + margin, y_max + margin, z_max], dtype=np.float32)
        self.register_buffer('grid_min', torch.from_numpy(grid_min))
        self.register_buffer('grid_max', torch.from_numpy(grid_max))

        occ_grid = self._build_occupancy_grid(map_points)
        self.register_buffer('occ_grid', occ_grid)
        n_occ = occ_grid.sum().item()
        print(f"[LOSEncoder] Grid {tuple(occ_grid.shape)}, "
              f"occupied={n_occ}/{occ_grid.numel()} ({n_occ/occ_grid.numel()*100:.1f}%)")

    def _build_occupancy_grid(self, map_points):
        pts = torch.from_numpy(map_points).float()
        idx = ((pts - self.grid_min) / self.voxel_size).long()
        grid_shape = ((self.grid_max - self.grid_min) / self.voxel_size).long() + 1
        Gx, Gy, Gz = grid_shape[0].item(), grid_shape[1].item(), grid_shape[2].item()
        idx[:, 0] = idx[:, 0].clamp(0, Gx - 1)
        idx[:, 1] = idx[:, 1].clamp(0, Gy - 1)
        idx[:, 2] = idx[:, 2].clamp(0, Gz - 1)
        occ = torch.zeros(Gx, Gy, Gz, dtype=torch.bool)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return occ

    def forward(self, xyz_query):
        B = xyz_query.shape[0]
        device = xyz_query.device
        start = self.bs_position.unsqueeze(0).expand(B, -1)
        end = xyz_query
        direction = end - start
        total_dist = torch.norm(direction, dim=-1)
        direction = direction / (total_dist.unsqueeze(-1) + 1e-8)
        max_steps = min(int((total_dist.max().item() / self.step_size)) + 2, 2000)
        los = torch.ones(B, dtype=torch.bool, device=device)
        obs_dist = torch.full_like(total_dist, fill_value=self.grid_max.norm().item())

        for step in range(1, max_steps + 1):
            if not los.any():
                break
            t = step * self.step_size
            points = start + direction * t
            idx = ((points - self.grid_min) / self.voxel_size).long()
            Gx, Gy, Gz = self.occ_grid.shape
            in_bounds = (
                (idx[:, 0] >= 0) & (idx[:, 0] < Gx) &
                (idx[:, 1] >= 0) & (idx[:, 1] < Gy) &
                (idx[:, 2] >= 0) & (idx[:, 2] < Gz)
            )
            check_mask = in_bounds & los
            if not check_mask.any():
                continue
            ci = idx[check_mask]
            occupied = self.occ_grid[ci[:, 0], ci[:, 1], ci[:, 2]]
            newly_blocked = check_mask.clone()
            newly_blocked[check_mask] = occupied
            obs_dist[newly_blocked] = t
            los[newly_blocked] = False

        los_dist = obs_dist / (total_dist + 1e-8)
        los_feat = torch.stack([los.float(), los_dist.clamp(0, 1)], dim=-1)
        return los_feat


# ========== 几何特征编码器 (替代 MapPointFeature) ==========

class GeometricFeatureExtractor(nn.Module):
    """
    几何特征编码器：从原始 .ply 点云计算物理几何量

    核心理念: 地图应该提供可计算的几何先验，而非可学习的随机特征。
    用 scipy KD-tree 在原始终 2.4M 点云上做空间查询，提取:
      1. 多尺度局部密度 (Local density features)
      2. BS-UE 射线 NLOS 检测 (Cylinder ray query)
      3. 近邻各向异性 (Local anisotropy for edge/corner detection)

    Args:
        map_points:  (N_ply, 3) 原始地图点云 (世界坐标)
        feature_dim: 输出特征维度 (默认 32)
        knn_k:       KNN 近邻数 (默认 32)
        bs_position: 基站位置 [x,y,z] (默认 [50,0,25])
        ray_width:   射线圆柱体半径 (米, 默认 0.5)
    """
    def __init__(self, map_points, feature_dim=32, knn_k=32,
                 bs_position=None, ray_width=0.5):
        super().__init__()
        from scipy.spatial import cKDTree

        self.feature_dim = feature_dim
        self.knn_k = knn_k
        self.ray_width = ray_width

        if bs_position is None:
            bs_position = np.array([50.0, 0.0, 25.0], dtype=np.float32)
        self.register_buffer('bs_pos', torch.from_numpy(np.array(bs_position, dtype=np.float32)))

        # 在原始点云上构建 KD-tree (点太少的区域只做随机采样子集)
        self.map_pts_np = map_points.astype(np.float64)
        self.kdtree = cKDTree(self.map_pts_np)
        print(f"[GeoFeature] KD-tree built on {len(self.map_pts_np)} points, "
              f"K={knn_k}, feat_dim={feature_dim}, ray_width={ray_width}m")

        # 小型 MLP: 几何原始特征 (9 维) → feature_dim
        # 9 维 = 3 密度 + 最近点距离 + BS距离 + NLOS遮挡距离 + 近邻Z方差 + 各向异性 + 中点密度
        self.geo_mlp = nn.Sequential(
            nn.Linear(9, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        # 轻量初始化
        for m in self.geo_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

        print(f"[GeoFeature] geo_mlp params: {sum(p.numel() for p in self.geo_mlp.parameters()):,}")

    def _ray_nlos_query(self, query_world):
        """
        圆柱体射线查询: 检查 BS→UE 射线是否被点云遮挡

        从 BS 沿 UE 方向步进，在每个位置检查以射线为轴的圆柱体内
        是否有地图点。这直接编码了"直射路径是否被建筑阻挡"的物理先验。

        Args:
            query_world: (B, 3) 查询位置 (世界坐标)
        Returns:
            nlos:        (B,) float: 首遮挡距离/总距离 (0=无遮挡, 1=完全遮挡)
            n_obstacles: (B,) float: 遮挡物数量 (归一化)
        """
        B = query_world.shape[0]
        device = query_world.device
        bs_np = self.bs_pos.cpu().numpy().astype(np.float64)

        nlos_list = []
        n_obs_list = []

        for b in range(B):
            ue_np = query_world[b].cpu().numpy().astype(np.float64)
            direction = ue_np - bs_np
            total_dist = np.linalg.norm(direction)
            if total_dist < 0.1:
                nlos_list.append(0.0)
                n_obs_list.append(0.0)
                continue

            direction = direction / total_dist
            step = 1.0  # 1m 步进
            n_steps = int(total_dist / step)

            first_hit = total_dist
            n_obstacles = 0
            last_hit = -10.0  # 防止同一障碍物被多次计数

            for s in range(1, n_steps):
                t = s * step
                pt = bs_np + direction * t
                # 查询圆柱体内的点数 (半径 = ray_width)
                count = len(self.kdtree.query_ball_point(pt, self.ray_width))
                if count > 5:  # 至少 6 个点才算障碍物
                    if first_hit > total_dist - 0.1:
                        first_hit = t
                    if t - last_hit > 3.0:  # 间隔 > 3m 算不同障碍物
                        n_obstacles += 1
                    last_hit = t

            nlos_list.append(min(first_hit / (total_dist + 1e-8), 1.0))
            n_obs_list.append(min(n_obstacles / 10.0, 1.0))  # 归一化

        nlos = torch.tensor(nlos_list, dtype=torch.float32, device=device)
        n_obs = torch.tensor(n_obs_list, dtype=torch.float32, device=device)
        return nlos, n_obs

    def forward(self, query_world):
        """
        Args:
            query_world: (B, 3) 查询位置 (世界坐标, 非归一化)
        Returns:
            geo_feat:    (B, feature_dim)
        """
        B = query_world.shape[0]
        device = query_world.device
        q_np = query_world.detach().cpu().numpy().astype(np.float64)

        # ---- KNN 查询 ----
        knn_dist, knn_idx = self.kdtree.query(q_np, k=self.knn_k)
        knn_pts = self.map_pts_np[knn_idx]  # (B, K, 3)

        # ---- 原始几何特征 (9 维) ----

        # 1-3. 多尺度密度: 0-5m, 5-20m, 20-50m 内的点数
        density_5m = np.array([len(self.kdtree.query_ball_point(q, 5.0)) for q in q_np])
        density_20m = np.array([len(self.kdtree.query_ball_point(q, 20.0)) for q in q_np])
        density_50m = np.array([len(self.kdtree.query_ball_point(q, 50.0)) for q in q_np])
        # 密度归一化 (取 log1p)
        d5 = np.log1p(density_5m) / np.log1p(500)   # 粗略上限
        d20 = np.log1p(density_20m) / np.log1p(5000)
        d50 = np.log1p(density_50m) / np.log1p(50000)

        # 4. 最近点距离 (m)
        nearest_dist = knn_dist[:, 0]  # (B,)
        nd = np.clip(nearest_dist / 50.0, 0, 1)  # 归一化

        # 5. BS 距离
        bs_np = self.bs_pos.cpu().numpy().astype(np.float64)
        bs_dist = np.linalg.norm(q_np - bs_np, axis=1)
        bd = np.clip(bs_dist / 500.0, 0, 1)  # 归一化

        # 6-7. NLOS 射线检测
        nlos, n_obs = self._ray_nlos_query(query_world)

        # 8. 近邻 Z 方差 (建筑高度变化)
        z_vals = knn_pts[:, :, 2]  # (B, K)
        z_std = np.std(z_vals, axis=1)
        zs = np.clip(z_std / 20.0, 0, 1)

        # 9. 近邻各向异性 (PCA ratio: 是否在建筑边缘)
        aniso = np.zeros(B, dtype=np.float32)
        for b in range(B):
            pts = knn_pts[b]  # (K, 3)
            pts_centered = pts - pts.mean(axis=0)
            cov = pts_centered.T @ pts_centered / (self.knn_k - 1)
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals[-1] > 1e-6:
                aniso[b] = eigvals[-1] / (eigvals.sum() + 1e-8)
        an = np.clip(aniso, 0, 1)

        # ---- 拼接 9 维几何特征 ----
        geo_raw = np.stack([d5, d20, d50, nd, bd, nlos.cpu().numpy(), n_obs.cpu().numpy(), zs, an], axis=1)
        geo_raw = torch.from_numpy(geo_raw.astype(np.float32)).to(device)  # (B, 9)

        # ---- 小型 MLP → feature_dim ----
        geo_feat = self.geo_mlp(geo_raw)  # (B, feature_dim)
        return geo_feat
