# -*- coding: utf-8 -*-
"""
信道解码器：将高斯场景表征解码为 MIMO-OFDM 信道矩阵

替代 WRF-GS+ 中的图像渲染器，采用可微特征聚合 + MLP 解码。

架构:
  1. 查询位置 → 位置编码 (Positional Encoding)
  2. 高斯特征聚合 (距离加权) → 场景感知特征
  3. 地图特征编码 (来自 MapPointFeature) → 环境感知特征
  4. 聚合特征 → MLP 解码 → 信道矩阵 [256, 4, 192] (实部 + 虚部)

参考:
  - WRF-GS+ DeformNetwork: 提供 per-gaussian 形变
  - 3DGS 体渲染权重: 用距离权重代替阿尔法混合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general_utils import build_rotation


class GaussianFeatureAggregator(nn.Module):
    """
    高斯特征聚合器

    对每个查询位置，计算所有高斯点的贡献权重，然后加权聚合特征。

    权重设计:
      w_i = opacity_i * exp(-||xyz_query - xyz_deformed_i||² / (2 * scale_i²))

    这类似于 3DGS 中像素到高斯的投影权重，但在 3D 空间中直接计算。
    """

    def __init__(self):
        super().__init__()

    def forward(self, xyz_query, xyz_gaussian, opacity, scaling, rotation, features,
                d_scaling=None, d_rotation=None):
        """
        Args:
            xyz_query:   (B, 3) 查询位置
            xyz_gaussian: (N, 3) 高斯中心（形变后）
            opacity:     (N, 1)
            scaling:     (N, 3) 各向异性尺度 (exp 激活后)
            rotation:    (N, 4) 旋转四元数 (归一化后)
            features:    (N, C_feat)
            d_scaling:   (N, 3) 或 None, DeformNetwork 尺度偏移
            d_rotation:  (N, 4) 或 None, DeformNetwork 旋转偏移

        Returns:
            agg_feat:    (B, C_feat) 聚合特征
            weights:     (B, N) 聚合权重
        """
        B = xyz_query.shape[0]
        N = xyz_gaussian.shape[0]
        C = features.shape[-1]

        # 调制尺度与旋转
        s_mod = scaling + 0.1 * d_scaling if d_scaling is not None else scaling
        r_mod = rotation + 0.1 * d_rotation if d_rotation is not None else rotation

        # 构建协方差矩阵 (各向异性): Cov = R * diag(s²) * R^T
        # Mahalanobis 距离: d² = (q-c)^T * Cov^(-1) * (q-c)
        # 等效于: d² = ||R^T * (1/s) * (q-c)||²  (白化变换)
        R = build_rotation(r_mod)  # (N, 3, 3)
        inv_s = 1.0 / (s_mod + 1e-8)  # (N, 3)

        # 每对 (b, n) 的加权距离: (B, N)
        diff = xyz_query.unsqueeze(1) - xyz_gaussian.unsqueeze(0)  # (B, N, 3)
        # 白化: (B, N, 3) → R^T @ diag(1/s) @ diff
        diff_rot = torch.einsum('nij,bnj->bni', R, diff)  # (B, N, 3): R^T @ diff
        diff_white = diff_rot * inv_s.unsqueeze(0)         # (B, N, 3): diag(1/s)
        mahal_dist2 = (diff_white ** 2).sum(dim=-1)        # (B, N)

        # 高斯权重
        w = opacity.squeeze(-1).unsqueeze(0).expand(B, -1)  # (B, N)
        w = w * torch.exp(-0.5 * mahal_dist2)

        # 归一化
        w_sum = w.sum(dim=-1, keepdim=True) + 1e-8
        w_norm = w / w_sum

        # 聚合
        feat = features.unsqueeze(0).expand(B, -1, -1)  # (B, N, C)
        agg_feat = torch.bmm(w_norm.unsqueeze(1), feat).squeeze(1)  # (B, C)

        return agg_feat, w_norm


class ChannelDecoder(nn.Module):
    """
    信道解码器：聚合特征 → MLP → MIMO-OFDM 信道

    MLP 架构:
      输入: C_agg (位置特征 + 高斯聚合特征 + 地图特征)
      → 隐藏层: [1024, 1024, 512, 512] (ReLU + LayerNorm)
      → 输出分支:
          - H_real: (256, 4, 192)
          - H_imag: (256, 4, 192)
      → 合并为复数: H = H_real + 1j * H_imag
    """

    def __init__(self, input_dim, hidden_dims=None, output_shape=(256, 4, 192)):
        """
        Args:
            input_dim:  聚合特征维度
            hidden_dims: MLP 隐藏层维度列表
            output_shape: 输出信道形状 (bs_ant, ue_ant, subcarrier)
        """
        super().__init__()
        self.output_shape = output_shape
        out_features = output_shape[0] * output_shape[1] * output_shape[2]

        if hidden_dims is None:
            hidden_dims = [1024, 1024, 512, 512]

        # MLP 编码器
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.05),  # 轻量 dropout 防过拟合
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        # 输出分支: 实部和虚部
        self.head_real = nn.Linear(prev_dim, out_features)
        self.head_imag = nn.Linear(prev_dim, out_features)

        # Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 可学习的输出缩放因子 — 匹配信道数据的量级 (约 1e-4)
        self.output_log_scale = nn.Parameter(torch.tensor(-8.0))  # exp(-8) ≈ 3e-4

        print(f"[ChannelDecoder] input_dim={input_dim}, output={output_shape}, "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) 聚合特征

        Returns:
            h_real: (B, 256, 4, 192)
            h_imag: (B, 256, 4, 192)
        """
        feat = self.encoder(x)  # (B, hidden[-1])

        # 输出实部和虚部
        h_real = self.head_real(feat)  # (B, 256*4*192)
        h_imag = self.head_imag(feat)

        # Reshape
        B = x.shape[0]
        h_real = h_real.view(B, *self.output_shape)
        h_imag = h_imag.view(B, *self.output_shape)

        # 应用可学习输出缩放 (匹配真实信道量级 ~1e-4)
        scale = torch.exp(self.output_log_scale)
        h_real = h_real * scale
        h_imag = h_imag * scale

        return h_real, h_imag
