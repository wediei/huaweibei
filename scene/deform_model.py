import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pos_utils import DeformNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    """
    可形变模型：对查询位置编码并输出每个高斯的形变参数

    比赛版本：支持输入地图特征 + 位置编码
    """
    def __init__(self, is_blender=True, is_6dof=False, map_feat_dim=0, gaussian_feat_dim=3):
        self.map_feat_dim = map_feat_dim
        self.gaussian_feat_dim = gaussian_feat_dim

        # 判断是否使用地图特征
        use_map = map_feat_dim > 0

        self.deform = DeformNetwork(
            is_blender=is_blender,
            is_6dof=is_6dof,
            use_map_feature=use_map,
            map_feat_dim=map_feat_dim,
            gaussian_feat_dim=gaussian_feat_dim
        ).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb, map_feat=None):
        """
        Args:
            xyz:      (N, 3) 高斯位置
            time_emb: (N, 3) 查询位置 (复用原接口名称)
            map_feat: (1, map_feat_dim) 或 (N, map_feat_dim) 地图特征 (可选)
        Returns:
            d_xyz:       (N, 3) 位置偏移
            d_rotation:  (N, 4) 旋转偏移
            d_scaling:   (N, 3) 尺度偏移
            d_signal:    (N, 3) 信号调制（特征调制）
        """
        return self.deform(xyz, time_emb, map_feat)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
