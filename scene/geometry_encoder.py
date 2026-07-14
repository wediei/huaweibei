# -*- coding: utf-8 -*-
"""
几何特征计算器：从地图点云计算 per-gaussian 几何特征

纯几何计算 — 零可学习参数。这是"真正的地图利用"的核心：
地图 XYZ 决定方向、距离、延迟，这些都是物理量。

BS 位置固定为 [50.0, 0.0, 25.0]。
光速 c = 3e8 m/s 用于计算路径延迟。
"""

import torch
import torch.nn as nn


class GeometricFeatureComputer(nn.Module):
    """
    计算每个 (查询位置, 高斯体) 对的几何特征。

    无任何可学习参数 — 所有输出都是地图几何的直接推导。

    输出:
        dir_dep:   (B, N, 3) 从 BS 到高斯的出发方向 (单位向量)
        dir_arr:   (B, N, 3) 从高斯到 UE 的到达方向 (单位向量)
        d_bs:      (B, N, 1) 高斯到 BS 的距离
        d_ue:      (B, N, 1) 高斯到 UE 的距离
        tau:       (B, N, 1) 总路径延迟 (秒)
        cos_scat:  (B, N, 1) 散射角余弦
        az_dep:    (B, N)    出发方位角 (弧度, [-π, π])
        el_dep:    (B, N)    出发仰角 (弧度, [-π/2, π/2])
    """

    def __init__(self, bs_position=(50.0, 0.0, 25.0), speed_of_light=3e8):
        super().__init__()
        self.register_buffer(
            'bs_pos',
            torch.tensor(bs_position, dtype=torch.float).view(1, 1, 3)
        )
        self.c = speed_of_light

    def forward(self, gauss_xyz, query_ue):
        """
        Args:
            gauss_xyz: (N, 3) 高斯位置 (世界坐标, 米)
            query_ue:  (B, 3) 查询 UE 位置 (世界坐标, 米)

        Returns:
            dict: 所有几何特征
        """
        N = gauss_xyz.shape[0]
        B = query_ue.shape[0]

        # 广播: (1, N, 3) 和 (B, 1, 3)
        g = gauss_xyz.unsqueeze(0)   # (1, N, 3)
        u = query_ue.unsqueeze(1)    # (B, 1, 3)

        # ---- 出发方向 (BS → 高斯) ----
        vec_dep = g - self.bs_pos               # (1/B, N, 3)
        d_bs = torch.norm(vec_dep, dim=-1, keepdim=True)  # (1/B, N, 1)
        d_bs = d_bs.clamp(min=0.01)             # 防止除零 (高斯在 BS 上)
        dir_dep = vec_dep / d_bs                # (1/B, N, 3)

        # 出发角
        az_dep = torch.atan2(dir_dep[..., 1], dir_dep[..., 0])  # (1/B, N)
        el_dep = torch.asin(dir_dep[..., 2].clamp(-1, 1))       # (1/B, N)

        # ---- 到达方向 (高斯 → UE) ----
        vec_arr = u - g                         # (B, N, 3)
        d_ue = torch.norm(vec_arr, dim=-1, keepdim=True)  # (B, N, 1)
        d_ue = d_ue.clamp(min=0.01)
        dir_arr = vec_arr / d_ue                # (B, N, 3)

        # ---- 路径延迟 ----
        tau = (d_bs + d_ue) / self.c            # (1/B, N, 1)

        # ---- 散射角 ----
        cos_scat = -(dir_dep * dir_arr).sum(dim=-1, keepdim=True)  # (B, N, 1)

        # ---- 对数距离 (给网络用的数值友好格式) ----
        log_d_bs = torch.log(d_bs + 1.0)        # (1/B, N, 1)
        log_d_ue = torch.log(d_ue + 1.0)        # (B, N, 1)

        return {
            'dir_dep':    dir_dep.expand(B, -1, -1),   # (B, N, 3)
            'dir_arr':    dir_arr,                     # (B, N, 3)
            'd_bs':       d_bs.expand(B, -1, -1),      # (B, N, 1)
            'd_ue':       d_ue,                        # (B, N, 1)
            'tau':        tau.expand(B, -1, -1),       # (B, N, 1)
            'cos_scat':   cos_scat,                    # (B, N, 1)
            'az_dep':     az_dep.expand(B, -1),        # (B, N)
            'el_dep':     el_dep.expand(B, -1),        # (B, N)
            'log_d_bs':   log_d_bs.expand(B, -1, -1),  # (B, N, 1)
            'log_d_ue':   log_d_ue,                    # (B, N, 1)
        }
