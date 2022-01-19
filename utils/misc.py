import threading

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc
import tqdm
from torch import cdist

from configs import TrainConfig

try:
    from open3d.cuda.pybind.geometry import PointCloud
    from open3d.cuda.pybind.utility import Vector3dVector
except (ModuleNotFoundError, ImportError):
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector
import random
from cmath import cos
from cmath import sin
from math import ceil, cos, sin
import open3d as o3d


def fp_sampling(points, num):
    batch_size = points.shape[0]
    D = cdist(points, points)
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    res = torch.zeros((batch_size, num), dtype=torch.int32, device=points.device)
    ds = D[:, 0, :]
    for i in range(1, num):
        idx = torch.argmax(ds, dim=1)
        res[:, i] = idx
        ds = torch.minimum(ds, D[torch.arange(batch_size), idx, :])

    return res


def create_cube():
    cube = []
    for _ in range(2500):
        p = np.random.rand((3)) - 0.5
        cube.append([-0.5, p[1], p[2]])
        cube.append([0.5, p[1], p[2]])
        cube.append([p[0], -0.5, p[2]])
        cube.append([p[0], 0.5, p[2]])
        cube.append([p[0], p[1], -0.5])
        cube.append([p[0], p[1], 0.5])

    cb = PointCloud()
    cb.points = Vector3dVector(np.array(cube))
    cb.paint_uniform_color([0, 1, 0])

    return cb


def create_sphere():
    sphere = []
    for _ in range(10000):
        sph_radius = 0.5
        y = random.uniform(-sph_radius, sph_radius)
        theta = random.uniform(0, 2 * np.pi)
        x = np.sqrt(sph_radius ** 2 - y ** 2) * cos(theta)
        z = np.sqrt(sph_radius ** 2 - y ** 2) * sin(theta)
        sphere.append([x, y, z])

    sph = PointCloud()
    sph.points = Vector3dVector(np.array(sphere))
    sph.paint_uniform_color([1, 0, 0])

    return sph


def sample_point_cloud(mesh, n_points=8192, dist=None, noise_rate=0.1, tolerance=0.01):
    """
    http://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html
    Produces input for implicit function
    :param mesh: Open3D mesh
    :param noise_rate: rate of gaussian noise added to the point sampled from the mesh
    :param percentage_sampled: percentage of point that must be sampled uniform
    :param total: total number of points that must be returned
    :param tollerance: maximum distance from mesh for a point to be considered 1.
    :param mode: str, one in ["unsigned", "signed", "occupancy"]
    :return: points (N, 3), occupancies (N,)
    """
    assert sum(dist) == 1.0

    n_uniform = int(n_points * dist[0])
    n_noise = int(n_points * dist[1])
    n_mesh = n_points - (n_uniform + n_noise)

    points_uniform = np.random.rand(n_uniform, 3) - 0.5
    points_noisy = np.array(mesh.sample_points_uniformly(n_noise, seed=TrainConfig.seed).points) + np.random.normal(0, noise_rate, (n_noise, 3))
    points_surface = np.array(mesh.sample_points_uniformly(n_mesh, seed=TrainConfig.seed).points)

    points = np.concatenate([points_uniform, points_noisy, points_surface], axis=0)

    if tolerance > 0:
        labels = check_mesh_contains([mesh], [points], tolerance=tolerance).squeeze().tolist()
    elif tolerance == 0:
        labels = [False] * (n_uniform + n_noise) + [True] * n_mesh

    return points, labels


def create_3d_grid(min_value=-0.5, max_value=0.5, step=0.04, batch_size=1):
    x_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    y_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    z_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    grid_2d = torch.cartesian_prod(x_range, y_range)
    grid_2d = grid_2d.repeat(x_range.shape[0], 1)
    z_repeated = z_range.unsqueeze(1).T.repeat(x_range.shape[0] ** 2, 1).T.reshape(-1)[..., None]
    grid_3d = torch.cat((grid_2d, z_repeated), dim=-1)
    grid_3d = grid_3d.unsqueeze(0).repeat(batch_size, 1, 1)
    return grid_3d


def check_mesh_contains(meshes, queries, tolerance=0.01):
    occupancies = []

    for mesh, query in zip(meshes, queries):
        scene = o3d.t.geometry.RaycastingScene()
        # mesh = o3d.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(o3d.pybind.core.Tensor(np.array(mesh.vertices, dtype=np.float32)),
                                o3d.pybind.core.Tensor(np.array(mesh.triangles, dtype=np.uint32)))
        query_points = o3d.pybind.core.Tensor(query.astype(np.float32))
        unsigned_distance = scene.compute_distance(query_points)
        occupancies.append((unsigned_distance.numpy() < tolerance))

    occupancies = np.stack(occupancies)[..., None]
    return occupancies


def from_depth_to_pc(depth, intrinsics, depth_factor=10000.):
    fx, fy, cx, cy = intrinsics
    points = []
    h, w = depth.shape
    for u in range(0, h):
        for v in range(0, w):
            z = depth[u, v]
            if z != 0:
                z = z / depth_factor
                x = ((v - cx) * z) / fx
                y = ((u - cy) * z) / fy
                points.append([x, y, z])
    points = np.array(points)
    return points


def project_pc(rgb, points):
    k = np.eye(3)
    k[0, :] = np.array([1066.778, 0, 312.9869])
    k[1, 1:] = np.array([1067.487, 241.3109])

    points = np.array(points) * 10000.0
    uv = k @ points.T
    uv = uv[0:2] / uv[2, :]

    uv = np.round(uv, 0).astype(int)

    uv[0, :] = np.clip(uv[0, :], 0, 639)
    uv[1, :] = np.clip(uv[1, :], 0, 479)

    rgb[uv[1, :], uv[0, :], :] = np.tile((np.array([1, 0, 0]) * 255).astype(int), (uv.shape[1], 1))

    return rgb


def project_onto_plane(x, n):
    d = np.dot(x, n) / np.linalg.norm(n)
    n = n / np.linalg.norm(n)
    p = d * n
    return x - p


def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
