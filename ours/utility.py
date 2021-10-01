import numpy as np
import torch
from open3d import open3d
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector
from open3d.open3d.visualization import draw_geometries
from dataclasses import dataclass
from torch import randn_like
import os
import open3d as o3d
import tqdm


# PARAMETERS ###########################################################################################################

device = "cuda"


@dataclass
class DataConfig:
    DATA_PATH = os.path.join("../data", "ShapeNet55-34", "ShapeNet-55")
    NAME = "ShapeNet"
    N_POINTS = 8192
    subset = "train"
    PC_PATH = os.path.join("../data", "ShapeNet55-34", "shapenet_pc")


@dataclass
class ModelConfig:
    NAME = "PoinTr"
    PC_SIZE = 2048
    knn_layer = 1
    num_pred = 6144
    device = device
    # Transformer
    n_channels = 3
    embed_dim = 384
    encoder_depth = 6
    mlp_ratio = 2.
    qkv_bias = False
    num_heads = 6
    attn_drop_rate = 0.
    drop_rate = 0.
    qk_scale = None
    out_size = 1024
    # Hypernetwork
    hidden_dim = 32


@dataclass
class TrainConfig:
    difficulty = "easy"
    device = device
    voxel_size = 0.1
    noise_rate = 0.1
    percentage_sampled = 0.1
    n_epoch = 5
    log_metrics_every = 100
    log_pcs_every = 10000

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

    # t = 0
    # f = 0
    # for elem in results:
    #     if elem:
    #         t += 1
    #     else:
    #         f += 1
    # print("Found ", t, " points inside the voxels and ", f, " points outside the voxel")

    return np.array(pcd.points), np.array(results)


def pc_grid_reconstruction(model, min_value=-1, max_value=1, step=0.05):
    x_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    y_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    z_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    grid_2d = torch.cartesian_prod(x_range, y_range)
    results = []
    for z in tqdm.tqdm(z_range):
        repeated_z = z.reshape(1, 1).repeat(grid_2d.shape[0], 1)
        grid_3d = torch.cat((grid_2d, repeated_z), dim=1)

        # TODO REMOVE DEBUG
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(grid_3d)
        # draw_geometries([pcd])
        # TODO END DEBUG

        result = model(grid_3d)
        result = torch.cat((grid_3d, result), dim=-1)
        good_ids = torch.nonzero(result[..., -1] == 1.).squeeze(1)
        result = result[good_ids]
        results.append(result[..., :-1])

    return torch.cat(results, dim=0)


# Test function
if __name__ == "__main__":

    def model(elem):
        import random
        if random.random() > 0.5:
            return torch.zeros_like(elem).fill_(1.)[:, :1]
        else:
            return torch.zeros_like(elem).fill_(0.)[:, :1]

    res = pc_grid_reconstruction(model)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(res)
    draw_geometries([pcd])
