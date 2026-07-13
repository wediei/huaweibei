# -*- coding: utf-8 -*-
"""
无线信道损失函数

实现大赛评价指标对应的可微损失：
  C = 0.4*C1 + 0.4*C2 + 0.2*(1/(1+C3))
  C1: PAS 余弦相似度 (Power Angular Spectrum)
  C2: PDP 余弦相似度 (Power Delay Profile)
  C3: NMSE 归一化均方误差

所有损失都以 "越小越好" 的方向设计，可直接相加。
"""

import torch
import torch.nn.functional as F
import torch.fft


def compute_pas(h, bs_antenna_dim=0):
    """
    计算基站侧功率角度谱 (PAS)

    PAS = sum_{sc, ue} |H[:, ue, sc]|²
    将子载波和用户天线维度聚合，得到每个基站天线的功率

    Args:
        h: (..., 256, 4, 192) 复数张量 (complex64/complex128)
        bs_antenna_dim: 基站天线维度索引（默认0）

    Returns:
        pas: (..., 256) 功率角度谱
    """
    # h shape: (batch, 256, 4, 192)
    pas = (h.abs() ** 2).sum(dim=(-1, -2))  # (batch, 256)
    return pas


def compute_pdp(h, subcarrier_dim=-1):
    """
    计算功率时延谱 (PDP)

    PDP = IFFT(H, dim=子载波) -> 对时域做功率累加
    将基站和用户天线维度聚合

    Args:
        h: (..., 256, 4, 192) 复数张量
        subcarrier_dim: 子载波维度（默认-1=192维）

    Returns:
        pdp: (..., 192) 功率时延谱
    """
    # 沿子载波维度做 IFFT
    h_time = torch.fft.ifft(h, n=h.shape[subcarrier_dim], dim=subcarrier_dim)
    # 聚合天线维度 -> 功率
    pdp = (h_time.abs() ** 2).sum(dim=(-2, -3))  # 聚合 BS 和 UE 天线: (batch, 192)
    return pdp


def compute_nmse(h_pred, h_gt, eps=1e-10):
    """
    归一化均方误差 NMSE

    NMSE = ||H_pred - H_gt||² / ||H_gt||²

    Args:
        h_pred: (..., 256, 4, 192) 预测信道（复数）
        h_gt:   (..., 256, 4, 192) 真实信道（复数）

    Returns:
        nmse: (,) 标量 NMSE
    """
    error = h_pred - h_gt
    mse = (error.abs() ** 2).sum()
    power = (h_gt.abs() ** 2).sum() + eps
    return mse / power


def cosine_similarity_loss(x, y):
    """
    余弦距离损失: 1 - cos_sim(x, y)

    Args:
        x, y: (batch, D) 特征向量

    Returns:
        loss: (,) 标量
    """
    # 归一化
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    cos_sim = (x_norm * y_norm).sum(dim=-1)  # (batch,)
    return (1.0 - cos_sim).mean()


class ChannelLoss(torch.nn.Module):
    """
    信道预测总损失

    对齐大赛评价指标：
      L_total = 0.4 * L_pas + 0.4 * L_pdp + 0.2 * L_nmse

    Args:
        w_pas:   PAS 损失权重 (默认 0.4)
        w_pdp:   PDP 损失权重 (默认 0.4)
        w_nmse:  NMSE 损失权重 (默认 0.2)
        use_real_imag: 如果 True, h_pred/h_gt 是 (batch, 2, 256, 4, 192)
                       其中 2 维度是 [real, imag], 需要合并为复数
        nmse_clip:  NMSE 裁剪上限，防止梯度爆炸 (默认 20.0)
    """

    def __init__(self, w_pas=0.4, w_pdp=0.4, w_nmse=0.2, use_real_imag=True, nmse_clip=20.0):
        super().__init__()
        self.w_pas = w_pas
        self.w_pdp = w_pdp
        self.w_nmse = w_nmse
        self.use_real_imag = use_real_imag
        self.nmse_clip = nmse_clip

    def _to_complex(self, h):
        """如果输入是 [real, imag] 分开的，合并为复数"""
        if self.use_real_imag:
            # h: (batch, 2, 256, 4, 192) -> complex
            return torch.complex(h[:, 0], h[:, 1])
        return h

    def forward(self, h_pred, h_gt):
        """
        Args:
            h_pred: (batch, 2, 256, 4, 192) 或 (batch, 256, 4, 192) complex
            h_gt:   与 h_pred 相同形状

        Returns:
            loss_total: 总损失 (标量)
            loss_dict:  各分量损失 dict
        """
        hp = self._to_complex(h_pred)
        hg = self._to_complex(h_gt)

        # PAS 损失
        pas_pred = compute_pas(hp)
        pas_gt = compute_pas(hg)
        l_pas = cosine_similarity_loss(pas_pred, pas_gt)

        # PDP 损失
        pdp_pred = compute_pdp(hp)
        pdp_gt = compute_pdp(hg)
        l_pdp = cosine_similarity_loss(pdp_pred, pdp_gt)

        # NMSE 损失 (带裁剪，防止梯度爆炸)
        l_nmse = compute_nmse(hp, hg)
        l_nmse_clipped = torch.clamp(l_nmse, max=self.nmse_clip)

        # 总损失
        loss_total = self.w_pas * l_pas + self.w_pdp * l_pdp + self.w_nmse * l_nmse_clipped

        loss_dict = {
            'loss_pas': l_pas.item(),
            'loss_pdp': l_pdp.item(),
            'loss_nmse': l_nmse.item(),
            'loss_nmse_clipped': l_nmse_clipped.item(),
            'loss_total': loss_total.item(),
        }

        return loss_total, loss_dict

    @torch.no_grad()
    def compute_metrics(self, h_pred, h_gt):
        """
        计算评价指标（用于验证，不反传梯度）

        Returns:
            metrics: dict {'pas_cos':, 'pdp_cos':, 'nmse':, 'score':}
        """
        hp = self._to_complex(h_pred)
        hg = self._to_complex(h_gt)

        # PAS
        pas_pred = compute_pas(hp)
        pas_gt = compute_pas(hg)
        pas_cos = (1.0 - cosine_similarity_loss(pas_pred, pas_gt)).item()

        # PDP
        pdp_pred = compute_pdp(hp)
        pdp_gt = compute_pdp(hg)
        pdp_cos = (1.0 - cosine_similarity_loss(pdp_pred, pdp_gt)).item()

        # NMSE
        nmse = compute_nmse(hp, hg).item()

        # 综合得分
        score = 0.4 * pas_cos + 0.4 * pdp_cos + 0.2 * (1.0 / (1.0 + nmse))

        return {
            'pas_cos': pas_cos,
            'pdp_cos': pdp_cos,
            'nmse': nmse,
            'score': score,
        }
