import open3d as o3d
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
import random
import time

from utils.misc import sample_point_cloud, sample_point_cloud2

#  0.1 0.4
def gen_box(min_side=0.1, max_side=0.4):
    sizes = []
    for i in range(3):
        sizes.append(random.uniform(min_side, max_side))
    cube_mesh = o3d.geometry.TriangleMesh.create_box(sizes[0], sizes[1], sizes[2])

    return cube_mesh


class BoxNet(data.Dataset):
    def __init__(self, config, n_samples, rotate=True):
        #  Backbone Input
        self.partial_points = config.partial_points
        self.multiplier_complete_sampling = config.multiplier_complete_sampling

        # Implicit function input
        self.noise_rate = config.noise_rate
        self.implicit_input_dimension = config.implicit_input_dimension
        self.tolerance = config.tolerance
        self.dist = config.dist

        # Synthetic dataset
        self.n_samples = n_samples
        self.rotate = rotate

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y
        # Find the mesh
        mesh = gen_box()

        while True:
            if self.rotate:
                rotation = R.random().as_matrix()
                mesh = mesh.rotate(rotation)

            # Define camera transformation and intrinsics
            #  (Camera is in the origin facing negative z, shifting it of z=1 puts it in front of the object)
            dist = 1.5
            complete_pcd = mesh.sample_points_uniformly(self.partial_points * self.multiplier_complete_sampling)
            _, pt_map = complete_pcd.hidden_point_removal([0, 0, dist], 1000)  # radius * 4
            partial_pcd = complete_pcd.select_by_index(pt_map)

            if len(np.array(partial_pcd.points)) != 0:
                break

        # Normalize the partial point cloud (all we could do at test time)
        partial_pcd = np.array(partial_pcd.points)
        mean = np.mean(np.array(partial_pcd), axis=0)
        partial_pcd = np.array(partial_pcd) - mean
        var = np.sqrt(np.max(np.sum(partial_pcd ** 2, axis=1)))

        partial_pcd = partial_pcd / (var * 2)
        # partial_pcd = partial_pcd / 2080  # TODO AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAA

        # Move the mesh so that it matches the partial point cloud position
        # (the [0, 0, 1] is to compensate for the fact that the partial pc is in the camera frame)
        mesh.translate(-mean)
        mesh.scale(1 / (var * 2), center=[0, 0, 0])

        # Sample labeled point on the mesh
        samples, occupancy = sample_point_cloud(mesh,
                                                n_points=self.implicit_input_dimension,
                                                dist=self.dist,
                                                noise_rate=self.noise_rate,
                                                tolerance=self.tolerance)

        partial_pcd = torch.FloatTensor(partial_pcd)

        # Set partial_pcd such that it has the same size of the others
        if partial_pcd.shape[0] > self.partial_points:
            perm = torch.randperm(partial_pcd.size(0))
            ids = perm[:self.partial_points]
            partial_pcd = partial_pcd[ids]
        else:
            print(f'Warning: had to pad the partial pcd - points {partial_pcd.shape[0]} added {self.partial_points - partial_pcd.shape[0]}')
            diff = self.partial_points - partial_pcd.shape[0]
            partial_pcd = torch.cat((partial_pcd, torch.zeros(diff, 3))) # TODO nooooooo

        samples = torch.tensor(samples).float()
        occupancy = torch.tensor(occupancy, dtype=torch.float)

        return 0, partial_pcd, [np.array(mesh.vertices), np.array(mesh.triangles)], samples, occupancy

    def __len__(self):
        return int(self.n_samples)


if __name__ == "__main__":
    from configs.local_config import DataConfig
    from tqdm import tqdm
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector

    a = DataConfig()
    a.dataset_path = Path("..", "data", "ShapeNetCore.v2")
    iterator = BoxNet(a)
    loader = DataLoader(iterator, num_workers=0, shuffle=False, batch_size=1)
    for elem in tqdm(loader):
        lab, part, mesh_vars, x, y = elem

        verts, tris = mesh_vars

        mesh = o3d.geometry.TriangleMesh(Vector3dVector(verts[0].cpu()), Vector3iVector(tris[0].cpu()))
        # o3d.visualization.draw_geometries([mesh], window_name="Complete")

        pc_part = PointCloud()
        pc_part.points = Vector3dVector(part[0])  # remove batch dimension
        # o3d.visualization.draw_geometries([pc], window_name="Partial")

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 1.5])
        o3d.visualization.draw_geometries([pc_part, mesh, coord], window_name="Both")

        pc = PointCloud()
        pc.points = Vector3dVector(x[0])  # remove batch dimension
        colors = []
        for i in y[0]:  # remove batch dimension
            if i == 0.:
                colors.append(np.array([1, 0, 0]))
            if i == 1.:
                colors.append(np.array([0, 1, 0]))
        colors = np.stack(colors)
        colors = Vector3dVector(colors)
        pc.colors = colors
        o3d.visualization.draw_geometries([pc, mesh])
