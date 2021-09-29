import numpy as np
import torch
from open3d import open3d
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector
from open3d.open3d.visualization import draw_geometries
from dataclasses import dataclass
from torch import randn_like
import os


# PARAMETERS ###########################################################################################################

device = "cuda"


@dataclass
class DataConfig:
    DATA_PATH = os.path.join("data", "ShapeNet55-34", "ShapeNet-55")
    NAME = "ShapeNet"
    N_POINTS = 8192
    subset = "test"
    PC_PATH = os.path.join("data", "ShapeNet55-34", "shapenet_pc")


@dataclass
class ModelConfig:
    NAME = "PoinTr"
    PC_SIZE = 2048
    knn_layer = 1
    num_pred = 6144
    num_query = 96
    trans_dim = 384
    device = device


@dataclass
class TrainConfig:
    difficulty = "easy"
    device = device
    voxel_size = 0.1
    noise_rate = 0.1
    percentage_sampled = 0.1

# USEFUL FUNCTION ######################################################################################################


def draw_point_cloud(x):
    x = x.cpu().squeeze()
    pc = PointCloud()
    pc.points = Vector3dVector(np.array(x))
    draw_geometries([pc])


crop_ratio = {
    'easy': 1/4,
    'median': 1/2,
    'hard': 3/4
}


def sample_point_cloud(xyz, voxel_size=0.1, noise_rate=0.1, percentage_sampled=0.1):  # 1 2048 3
    # TODO try also with https://blender.stackexchange.com/questions/31693/how-to-find-if-a-point-is-inside-a-mesh
    # VOXEL
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    voxel_grid = open3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    # open3d.open3d.visualization.draw_geometries([voxel_grid])

    # ORIGINAL PC
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(xyz)
    # open3d.open3d.visualization.draw_geometries([pcd])

    # WITH GAUSSIAN NOISE
    z = randn_like(xyz) * noise_rate
    xyz = xyz + z
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    # open3d.open3d.visualization.draw_geometries([pcd])

    # WITH 10% UNIFORM RANDOM POINTS
    k = int(len(xyz) * percentage_sampled)
    random_points = torch.FloatTensor(k, 3).uniform_(-1, 1)
    pcd = open3d.geometry.PointCloud()
    xyz = torch.cat((xyz, random_points))
    pcd.points = open3d.utility.Vector3dVector(xyz)
    # open3d.open3d.visualization.draw_geometries([pcd])

    # WITH LABEL
    results = voxel_grid.check_if_included(open3d.utility.Vector3dVector(xyz))
    colors = np.zeros((len(xyz), 3))
    for i, value in enumerate(results):
        if value:
            colors[i] = np.array([0, 0, 1])
        else:
            colors[i] = np.array([0, 1, 0])
    pcd.colors = open3d.utility.Vector3dVector(colors)
    # open3d.open3d.visualization.draw_geometries([pcd])

    t = 0
    f = 0
    for elem in results:
        if elem:
            t += 1
        else:
            f += 1
    print("Found ", t, " points inside the voxels and ", f, " points outside the voxel")

    return np.array(pcd.points), np.array(results)
