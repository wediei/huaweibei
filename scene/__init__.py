#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from torch.utils.data import DataLoader

# 条件导入: 仅在 Linux/CUDA 环境下可用的模块
try:
    import yaml
    from utils.system_utils import searchForMaxIteration
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from scene.gaussian_model import GaussianModel
    from arguments import ModelParams
    from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
    from scene.dataloader import *
    from scene.deform_model import DeformModel
except (ImportError, ModuleNotFoundError) as e:
    # Windows 开发环境: 这些模块依赖 CUDA 扩展 (simple_knn, diff_gaussian_rasterization, cv2)
    # 训练脚本直接导入所需模块, 不通过 scene 包
    print(f'[scene/__init__] Skipping CUDA-dependent imports: {e}')
    GaussianModel = None
    DeformModel = None
    sceneLoadTypeCallbacks = None
    cameraList_from_camInfos = None
    camera_to_JSON = None
    searchForMaxIteration = None
    ModelParams = None
    yaml = None


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.batch_size = 1
        self.datadir = "./data_test200" # Choose the dataset directory
        self.cameras_extent = 2

        yaml_file_path = os.path.join(self.datadir, 'gateway_info.yml')
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
        self.r_o = data['gateway1']['position']
        self.gateway_orientation = data['gateway1']['orientation']



        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        dataset = dataset_dict["rfid"]
        train_index = os.path.join(self.datadir, "train_index.txt")
        test_index = os.path.join(self.datadir, "test_index.txt")

        if not os.path.exists(train_index) or not os.path.exists(test_index):
            split_dataset(self.datadir, ratio=0.8, dataset_type="rfid")

        self.train_set = dataset(self.datadir, train_index)
        self.test_set = dataset(self.datadir, test_index)



        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def dataset_init(self):
        self.train_iter_dataset = iter(self.train_iter)
        self.test_iter_dataset = iter(self.test_iter)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


class Round1Scene:
    """
    Round1 比赛数据集 Scene 类
    管理数据加载、迭代
    """

    def __init__(self, datadir, gaussians=None, batch_size=4, shuffle=True, normalize_pos=True):
        self.datadir = datadir
        self.batch_size = batch_size
        self.gaussians = gaussians

        # 加载地图点云（用于编码器初始化）
        from scene.map_encoder import load_map_pointcloud
        self.map_points = load_map_pointcloud(datadir)  # (N_map, 3)

        # 创建数据集
        from scene.round1_dataset import Round1Dataset
        self.train_set = Round1Dataset(datadir, split='train', normalize_pos=normalize_pos)
        self.test_set = Round1Dataset(datadir, split='test', normalize_pos=normalize_pos,
                                      pos_mean=self.train_set.pos_mean,
                                      pos_std=self.train_set.pos_std)

        # DataLoader
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=0)

        # 场景范围（用于高斯初始化）
        pos = self.train_set.positions
        self.cameras_extent = np.max(pos.max(axis=0) - pos.min(axis=0))
        self.pos_mean = self.train_set.pos_mean
        self.pos_std = self.train_set.pos_std

        # 保存模型路径
        self.model_path = None

        print(f"Round1Scene initialized:")
        print(f"  Train samples: {len(self.train_set)}")
        print(f"  Test samples: {len(self.test_set)}")
        print(f"  Position range: X=[{pos[:,0].min():.1f}, {pos[:,0].max():.1f}], "
              f"Y=[{pos[:,1].min():.1f}, {pos[:,1].max():.1f}]")
        print(f"  Cameras extent: {self.cameras_extent:.1f}")

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def save(self, iteration, model_path):
        """保存高斯模型"""
        os.makedirs(model_path, exist_ok=True)
        if self.gaussians is not None:
            self.gaussians.save_ply(os.path.join(model_path, f"point_cloud_{iteration}.ply"))
