# -*- coding: utf-8 -*-
"""
Round1 无线信道数据集加载器
加载 .npy 格式的位置坐标和复数信道数据
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class Round1Dataset(Dataset):
    """
    无线信道大赛 Round1 数据集

    数据说明:
        - Train_Pos.npy:    (2000, 3) float64, 训练位置 (X, Y, Z)
        - Train_Channel.npy: (2000, 256, 4, 192) complex64, 训练信道
        - Test_Pos.npy:     (500, 3) float64, 测试位置

    输出:
        - pos:      (3,) float32, 归一化后的位置
        - ch_real:  (256, 4, 192) float32, 信道实部
        - ch_imag:  (256, 4, 192) float32, 信道虚部
    """

    def __init__(self, data_dir, split='train', normalize_pos=True, pos_mean=None, pos_std=None):
        """
        Args:
            data_dir:   数据目录 (Round1_Map/)
            split:      'train' 或 'test'
            normalize_pos: 是否对位置做标准化
            pos_mean:   预计算的位置均值 (None 则从训练集计算)
            pos_std:    预计算的位置标准差 (None 则从训练集计算)
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.normalize_pos = normalize_pos

        # 加载位置数据
        if split == 'train':
            pos_path = os.path.join(data_dir, 'Round1_Train_Pos.npy')
            ch_path = os.path.join(data_dir, 'Round1_Train_Channel.npy')
        else:
            pos_path = os.path.join(data_dir, 'Round1_Test_Pos.npy')
            ch_path = None  # 测试集没有 Channel

        self.positions = np.load(pos_path).astype(np.float32)  # (N, 3)
        self.n_samples = self.positions.shape[0]

        # 位置标准化
        if normalize_pos:
            if pos_mean is not None and pos_std is not None:
                self.pos_mean = np.array(pos_mean, dtype=np.float32)
                self.pos_std = np.array(pos_std, dtype=np.float32)
            else:
                # 从训练集计算统计量
                train_pos = np.load(os.path.join(data_dir, 'Round1_Train_Pos.npy')).astype(np.float32)
                self.pos_mean = train_pos.mean(axis=0)
                self.pos_std = train_pos.std(axis=0)
                self.pos_std = np.clip(self.pos_std, 1e-6, None)  # 防除零

        # 加载信道数据 (仅训练集)
        if split == 'train':
            print(f"Loading channel data from {ch_path} ...")
            # 使用 mmap 模式加载大文件
            self.channels = np.load(ch_path, mmap_mode='r')  # (2000, 256, 4, 192) complex64
            # 分离实部虚部
            self.ch_real = np.array(self.channels.real, dtype=np.float32)  # (2000, 256, 4, 192)
            self.ch_imag = np.array(self.channels.imag, dtype=np.float32)
            print(f"  Channel shape: {self.channels.shape}, dtype={self.channels.dtype}")
            # 释放 mmap
            del self.channels
        else:
            self.ch_real = None
            self.ch_imag = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 位置
        pos = self.positions[idx]  # (3,)
        if self.normalize_pos:
            pos = (pos - self.pos_mean) / self.pos_std

        pos_tensor = torch.from_numpy(pos).float()

        if self.split == 'train':
            # 信道 (实部 + 虚部)
            ch_r = torch.from_numpy(self.ch_real[idx]).float()   # (256, 4, 192)
            ch_i = torch.from_numpy(self.ch_imag[idx]).float()   # (256, 4, 192)
            return pos_tensor, ch_r, ch_i
        else:
            return pos_tensor

    def get_pos_stats(self):
        """返回位置标准化参数"""
        return self.pos_mean.copy(), self.pos_std.copy()
