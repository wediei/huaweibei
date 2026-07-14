import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10,
                 is_blender=True, is_6dof=False,
                 use_map_feature=False, map_feat_dim=0, gaussian_feat_dim=3,
                 geo_feat_dim=0):
        """
        可形变网络

        Args:
            D: MLP 深度
            W: MLP 宽度
            input_ch: 输入坐标维度 (3 for xyz)
            output_ch: 输出通道数 (未使用，改为固定分支)
            multires: 位置编码频率倍数
            is_blender: 使用 Blender 模式（单独的位置编码MLP）
            is_6dof: 使用 SE3 刚体变换
            use_map_feature: 是否使用地图特征
            map_feat_dim: 地图特征维度
            gaussian_feat_dim: 高斯特征调制维度 (d_signal)
            geo_feat_dim: 几何特征维度 (dir_dep, dir_arr, distances 等, 0=禁用)
        """
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.gaussian_feat_dim = gaussian_feat_dim
        self.is_blender = is_blender
        self.is_6dof = is_6dof
        self.use_map_feature = use_map_feature
        self.map_feat_dim = map_feat_dim
        self.geo_feat_dim = geo_feat_dim

        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        # 位置编码
        self.embed_pos_fn, pos_input_ch = get_embedder(self.t_multires, 3)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)

        # 总输入维度 = xyz_pe + pos_pe + (可选) map_feat + (可选) geo_feat
        self.total_input_ch = xyz_input_ch + pos_input_ch
        if use_map_feature and map_feat_dim > 0:
            self.total_input_ch += map_feat_dim
        if geo_feat_dim > 0:
            self.total_input_ch += geo_feat_dim

        if is_blender:
            self.pos_out = 90

            self.posnet = nn.Sequential(
                nn.Linear(pos_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.pos_out))

            # 最终输入维度: xyz_pe + pos_pe + (可选的) map_feat + (可选) geo_feat
            final_input_ch = xyz_input_ch + self.pos_out
            if use_map_feature and map_feat_dim > 0:
                final_input_ch += map_feat_dim
            if geo_feat_dim > 0:
                final_input_ch += geo_feat_dim

            self.linear = nn.ModuleList(
                [nn.Linear(final_input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + final_input_ch, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.total_input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.total_input_ch, W)
                    for i in range(D - 1)]
            )

        # 输出分支
        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)
        self.gaussian_signal = nn.Linear(W, gaussian_feat_dim)
        # 移除了 gaussian_phase (复数信号调制)

    def forward(self, x, t, map_feat=None, geo_feat=None):
        """
        Args:
            x: (N, 3) 高斯位置
            t: (N, 3) 查询位置 (query_pos)
            map_feat: (N, map_feat_dim) 或 None 地图特征
            geo_feat: (N, geo_feat_dim) 或 None 几何特征 (方向、距离等)
        Returns:
            d_xyz: (N, 3)
            rotation: (N, 4)
            scaling: (N, 3)
            signal: (N, gaussian_feat_dim)
        """
        t_emb = self.embed_pos_fn(t)
        x_emb = self.embed_fn(x)

        # 快捷函数: 拼接所有特征
        def _cat_feats(base_list):
            parts = base_list[:]
            if map_feat is not None and self.use_map_feature:
                parts.append(map_feat)
            if geo_feat is not None and self.geo_feat_dim > 0:
                parts.append(geo_feat)
            return torch.cat(parts, dim=-1)

        if self.is_blender:
            t_emb = self.posnet(t_emb)

            h = _cat_feats([x_emb, t_emb])

            for i, l in enumerate(self.linear):
                h = self.linear[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = _cat_feats([x_emb, t_emb, h])
        else:
            h = _cat_feats([x_emb, t_emb])

            for i, l in enumerate(self.linear):
                h = self.linear[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = _cat_feats([x_emb, t_emb, h])

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)

        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)
        signal = self.gaussian_signal(h)  # (N, gaussian_feat_dim)

        return d_xyz, rotation, scaling, signal
