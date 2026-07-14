# -*- coding: utf-8 -*-
"""
几何接地解码器：用几何投影替代 MLP 解码器

核心思想 (区别于 R2/v4 的纯 MLP):
  1. 每个高斯的贡献按其几何属性 (方向→角度, 路径→延迟) 软投影到输出 bin
  2. 高斯的"散射向量" (SH 特征 + 不透明度) 被投影到 256 角度 bin 和 192 延迟 bin
  3. 投影后的 bin 特征 → 小 MLP → 因子矩阵 → einsum 重建完整信道

对比:
  v4: 107-dim bottleneck → MLP → Linear(256, 196608) → CP 因子
  v5: (B,N,28) 散射向量 → 软投影到 256/192 bin → Linear(128,R) → CP 因子

几何提供了"哪个高斯影响哪个 bin"的先验，大幅减少学习负担。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryGroundedDecoder(nn.Module):
    """
    几何接地解码器

    流程:
      1. 共享 MLP: 高斯散射向量 → 128-dim 潜在特征
      2. 角度投影: 根据出发方位角 θ 软分配 → 256 bins → Linear → (256, R) factor
      3. 延迟投影: 根据路径延迟 τ 软分配 → 192 bins → Linear → (192, R) factor
      4. UE 因子: 全局池化 → Linear → (4, R) factor
      5. 重建: einsum('bar,bjr,bsr->bajs', ant, ue, sc) * scale

    Args:
        scatter_dim:   散射向量维度 (SH特征 + 不透明度等)
        hidden_dim:    共享 MLP 隐藏维度
        rank:          CP 分解秩数
        angle_bins:    角度 bin 数 (默认 256 = BS 天线数)
        delay_bins:    延迟 bin 数 (默认 192 = 子载波数)
        ue_ants:       UE 天线数 (默认 4)
        tau_min:       最小延迟 (秒), 用于 bin 范围
        tau_max:       最大延迟 (秒)
    """

    def __init__(self, scatter_dim=28, hidden_dim=128, rank=16,
                 angle_bins=256, delay_bins=192, ue_ants=4,
                 tau_min=1e-8, tau_max=2e-6):
        super().__init__()
        self.rank = rank
        self.angle_bins = angle_bins
        self.delay_bins = delay_bins
        self.ue_ants = ue_ants

        # ---- 角度 bin 中心 (均匀分布 [-π, π]) ----
        angle_edges = torch.linspace(-torch.pi, torch.pi, angle_bins + 1)
        angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
        self.register_buffer('angle_centers', angle_centers)  # (256,)

        # ---- 延迟 bin 中心 (均匀分布 [tau_min, tau_max]) ----
        delay_edges = torch.linspace(tau_min, tau_max, delay_bins + 1)
        delay_centers = 0.5 * (delay_edges[:-1] + delay_edges[1:])
        self.register_buffer('delay_centers', delay_centers)  # (192,)

        # ---- 共享特征提取 ----
        self.shared_mlp = nn.Sequential(
            nn.Linear(scatter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ---- 投影头 (共享权重跨 bin) ----
        out_dim = rank * 2  # 实部 + 虚部

        self.ant_proj_real = nn.Linear(hidden_dim, rank)
        self.ant_proj_imag = nn.Linear(hidden_dim, rank)
        self.sc_proj_real  = nn.Linear(hidden_dim, rank)
        self.sc_proj_imag  = nn.Linear(hidden_dim, rank)
        self.ue_proj_real  = nn.Linear(hidden_dim, ue_ants * rank)
        self.ue_proj_imag  = nn.Linear(hidden_dim, ue_ants * rank)

        # ---- 可学习的扩散参数 ----
        # 角度扩散 (弧度): 每个高斯在角度域的影响宽度
        self.log_angle_spread = nn.Parameter(torch.tensor(-1.0))
        # 延迟扩散 (秒): 每个高斯在延迟域的影响宽度
        self.log_delay_spread = nn.Parameter(torch.tensor(-16.0))

        # ---- 输出缩放 (同 v4) ----
        self.output_log_scale = nn.Parameter(torch.tensor(-8.0))

        # ---- 初始化 ----
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        total_p = sum(p.numel() for p in self.parameters())
        print(f"[GeometryGroundedDecoder] scatter_dim={scatter_dim}, "
              f"hidden={hidden_dim}, rank={rank}, params={total_p:,}")

    def _soft_project_1d(self, values, bin_centers, log_spread):
        """
        一维软投影: 将标量值软分配到最近的 bin。

        使用高斯核: w[i,k] ∝ exp(-(value_i - center_k)² / (2 * σ²))
        跨 bin 归一化 (每个高斯贡献总和为 1)。

        Args:
            values:      (B, N) 标量值 (方位角或延迟)
            bin_centers: (M,)   bin 中心
            log_spread:  scalar 可学习的 log(σ)

        Returns:
            weights: (B, N, M) 软分配权重
        """
        B, N = values.shape
        M = bin_centers.shape[0]
        sigma = torch.exp(log_spread).clamp(min=5e-3)

        # (B, N, M) = (B, N, 1) - (1, 1, M)
        diff = values.unsqueeze(-1) - bin_centers.view(1, 1, M)  # (B, N, M)
        logits = -diff.pow(2) / (2.0 * sigma ** 2 + 1e-8)

        # 跨 bin 的 softmax (每个高斯的权重和为 1)
        weights = torch.softmax(logits, dim=-1)  # (B, N, M)

        return weights

    def forward(self, scatter_vec, az_dep, tau, return_factors=False):
        """
        Args:
            scatter_vec: (B, N, scatter_dim) 每个高斯的散射向量
                         包含: 调制后的 SH 特征 + 不透明度 + 几何特征
            az_dep:      (B, N) 出发方位角 (弧度)
            tau:         (B, N, 1) 或 (B, N) 路径延迟 (秒)
            return_factors: 是否返回中间因子 (调试用)

        Returns:
            h_real, h_imag: 各 (B, 256, 4, 192)
            或 (h_real, h_imag, factors_dict) 当 return_factors=True
        """
        B, N, _ = scatter_vec.shape
        R = self.rank
        A = self.angle_bins   # 256
        S = self.delay_bins   # 192
        U = self.ue_ants      # 4

        if tau.dim() == 3:
            tau = tau.squeeze(-1)  # (B, N, 1) → (B, N)

        # ---- 1. 共享特征提取 ----
        latent = self.shared_mlp(scatter_vec)  # (B, N, hidden_dim)

        # ---- 2. 角度软投影 ----
        angle_w = self._soft_project_1d(az_dep, self.angle_centers,
                                        self.log_angle_spread)  # (B, N, 256)

        # 聚合: ant_feat[k,:] = Σ_i angle_w[b,i,k] * latent[b,i,:]
        ant_feat = torch.bmm(angle_w.transpose(1, 2), latent)  # (B, 256, hidden_dim)

        ant_real = self.ant_proj_real(ant_feat)  # (B, 256, R)
        ant_imag = self.ant_proj_imag(ant_feat)  # (B, 256, R)

        # ---- 3. 延迟软投影 ----
        delay_w = self._soft_project_1d(tau, self.delay_centers,
                                        self.log_delay_spread)  # (B, N, 192)

        sc_feat = torch.bmm(delay_w.transpose(1, 2), latent)  # (B, 192, hidden_dim)

        sc_real = self.sc_proj_real(sc_feat)  # (B, 192, R)
        sc_imag = self.sc_proj_imag(sc_feat)  # (B, 192, R)

        # ---- 4. UE 因子 (全局池化 + MLP) ----
        ue_feat = latent.mean(dim=1)  # (B, hidden_dim)
        ue_real = self.ue_proj_real(ue_feat).view(B, U, R)  # (B, 4, R)
        ue_imag = self.ue_proj_imag(ue_feat).view(B, U, R)  # (B, 4, R)

        # ---- 5. CP 重建 ----
        scale = torch.exp(self.output_log_scale)

        h_real = torch.einsum('bar,bjr,bsr->bajs', ant_real, ue_real, sc_real) * scale
        h_imag = torch.einsum('bar,bjr,bsr->bajs', ant_imag, ue_imag, sc_imag) * scale

        if return_factors:
            return h_real, h_imag, {
                'ant_real': ant_real, 'ant_imag': ant_imag,
                'ue_real':  ue_real,  'ue_imag':  ue_imag,
                'sc_real':  sc_real,  'sc_imag':  sc_imag,
                'angle_w':  angle_w,  'delay_w':  delay_w,
            }

        return h_real, h_imag
