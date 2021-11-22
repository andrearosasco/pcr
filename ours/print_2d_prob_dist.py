from cmath import cos, sin
from functools import reduce
from math import sqrt
from pathlib import Path

import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
import open3d as o3d
from models.HyperNetwork import HyperNetwork
from configs.local_config import ModelConfig
from torch import nn
from utils.misc import create_3d_grid, check_mesh_contains
from datasets.ShapeNetPOV import ShapeNet
from torch.utils.data import DataLoader
from configs.local_config import DataConfig, TrainConfig
import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import cv2


def f1(p, r):
    return 2*((p*r)/(p+r))

if __name__ == "__main__":
    # Reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed = TrainConfig.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(0)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    model = HyperNetwork(ModelConfig())

    model.load_state_dict(torch.load("checkpoint/server2.ptc"))
    model.cuda()
    model.eval()

    # TODO START MINE
    # Extract point cloud from mesh
    complete_path = 'microwave/model_normalized.obj'

    tm = o3d.io.read_triangle_mesh(complete_path, False)
    complete_pcd = tm.sample_points_uniformly(10000)

    # Get random position of camera
    sph_radius = 1
    y = random.uniform(-sph_radius, sph_radius)
    theta = random.uniform(0, 2 * np.pi)
    x = np.sqrt(sph_radius ** 2 - y ** 2) * cos(theta)
    z = np.sqrt(sph_radius ** 2 - y ** 2) * sin(theta)
    camera = [x, y, z]

    # Remove hidden points
    _, pt_map = complete_pcd.hidden_point_removal(camera, 500)  # radius * 4
    partial_pcd = complete_pcd.select_by_index(pt_map)
    # TODO END MINE
    # TODO remember sigmoid after layer
    grid = torch.cartesian_prod(torch.arange(-0.5, 0.5, 0.001), torch.arange(-0.5, 0.5, 0.001)).unsqueeze(0).cuda()
    zs = torch.zeros_like(grid)[..., :1]
    grid = torch.cat((grid, zs), dim=-1)

    with torch.no_grad():
        partial = torch.FloatTensor(np.array(partial_pcd.points)).unsqueeze(0).cuda()
        fast_weights, _ = model.backbone(partial)
        results = model.sdf(grid, fast_weights)
        res = torch.sigmoid(results[0])

        size = int(sqrt(res.shape[0]))
        res = res.reshape((size, size, 1))

        cv2.imshow("Prob", res.cpu().numpy())
        cv2.waitKey(10000)
        pass
