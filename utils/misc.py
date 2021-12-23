import math

import numpy as np
import torch
from torch import cdist
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import default_collate

try:
    from open3d.cuda.pybind.geometry import PointCloud
    from open3d.cuda.pybind.utility import Vector3dVector
except (ModuleNotFoundError, ImportError):
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector
import random
from cmath import cos
from cmath import sin
from math import cos, sin
import open3d as o3d


def collate(batch):
    """'
     Since every mesh in a batch is represented by two variable-length tensors
     we need to packed them so that the dataloader is able to return them.
     We can convert it to a padded tensor using pad_packed_sequence(x, batch_first=True, padding_value=0.)
    """
    transposed = zip(*batch)
    out = []
    for samples in transposed:
        if isinstance(samples[0], list):
            meshes = []
            for seq in list(zip(*samples)):
                packed = pack_sequence(list(seq), enforce_sorted=False)
                meshes.append(packed)
            out.append(meshes)
        else:
            out.append(default_collate(samples))
    return out


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
    www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html
    """
    assert sum(dist) == 1.0
    labels = None

    n_uniform = int(n_points * dist[0])
    n_noise = int(n_points * dist[1])
    n_mesh = n_points - (n_uniform + n_noise)

    points_uniform = np.random.rand(n_uniform, 3) - 0.5
    points_noisy = np.array(mesh.sample_points_uniformly(n_noise).points) + np.random.normal(0, noise_rate, (n_noise, 3))
    points_surface = np.array(mesh.sample_points_uniformly(n_mesh).points)

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


def fp_sampling(points, num, starting_point=None):
    batch_size = points.shape[0]
    # If no starting_point is provided, the starting point is the first point of points
    if starting_point is None:
        starting_point = points[:, 0].unsqueeze(1)
    D = cdist(starting_point, points).squeeze(1)

    perm = torch.zeros((batch_size, num), dtype=torch.int32, device=points.device)
    ds = D
    for i in range(0, num):
        idx = torch.argmax(ds, dim=1)
        perm[:, i] = idx
        ds = torch.minimum(ds, cdist(points[torch.arange(batch_size), idx].unsqueeze(1), points).squeeze())

    return perm


def project_onto_plane(x, n):
    d = np.dot(x, n) / np.linalg.norm(n)
    n = n / np.linalg.norm(n)
    p = d * n
    return x - p


def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
