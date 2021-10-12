import os
import random
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.utils.data as data
import open3d as o3d
from numpy import cos, sin
from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries, Visualizer

from utils.misc import sample_point_cloud




class ShapeNet(data.Dataset):
    def __init__(self, config):
        #  Backbone Input
        self.data_root = Path(config.dataset_path)
        self.mode = config.mode
        self.n_points = config.N_POINTS

        # Implicit function input
        self.noise_rate = config.noise_rate
        self.percentage_sampled = config.percentage_sampled

        with (self.data_root / f'{self.mode}.txt').open('r') as file:
            lines = file.readlines()

        self.samples = lines
        self.n_samples = len(lines)

        with (self.data_root / 'classes.txt').open('r') as file:
            self.labels_map = {l.split()[1]: l.split()[0] for l in file.readlines()}

        print("Found ", self.n_samples, " instances")

    @staticmethod
    def pc_norm(pcs):
        """ pc: NxC, return NxC """
        centroid = np.mean(pcs, axis=0)
        pcs = pcs - centroid
        m = np.max(np.sqrt(np.sum(pcs ** 2, axis=1)))
        pcs = pcs / m
        return pcs

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y
        # Get label
        dir_path = self.data_root / self.samples[idx].strip()
        label = self.labels_map[dir_path.parent.name]

        # Extract point cloud from mesh
        tm = o3d.io.read_triangle_mesh(str(dir_path / 'models/model_normalized.obj'), True)

        complete_pcd = tm.sample_points_uniformly(self.n_points*10)

        diameter = np.linalg.norm(
            np.asarray(complete_pcd.get_max_bound()) - np.asarray(complete_pcd.get_min_bound()))

        z = random.uniform(-diameter, diameter)
        theta = random.uniform(0, 2*np.pi)
        x = ((diameter)**2 - z**2) * cos(theta)
        y = ((diameter)**2 - z**2) * sin(theta)

        camera = [x, y, z]
        f = TriangleMesh.create_coordinate_frame(size=1, origin=camera)

        radius = np.sqrt(diameter**2 - z**2) * 100

        _, pt_map = complete_pcd.hidden_point_removal(camera, radius*4)
        partial_pcd = complete_pcd.select_by_index(pt_map)

        # tm.compute_vertex_normals()
        # draw_geometries([tm])
        draw_geometries([partial_pcd], lookat=[0, 0, 0], up=[0, 1, 0], front=camera, zoom=1)

        partial_pcd = np.array(partial_pcd.points)
        # partial_pcd = self.pc_norm(partial_pcd)
        partial_pcd = torch.FloatTensor(partial_pcd)
        complete_pcd = np.array(complete_pcd.points)
        # complete_pcd = self.pc_norm(complete_pcd)
        complete_pcd = torch.FloatTensor(complete_pcd)

        imp_x, imp_y = sample_point_cloud(tm,
                                          self.noise_rate,
                                          self.percentage_sampled)
        imp_x, imp_y = torch.tensor(imp_x).float(), torch.tensor(imp_y).bool().float().bool().float()  # TODO oh god..

        # return label, (complete_xyz, complete_colors), imp_x, imp_y
        return label, complete_pcd, partial_pcd, imp_x, imp_y

    def __len__(self):
        return self.n_samples



if __name__ == "__main__":
    from configs.cfg1 import DataConfig
    iterator = ShapeNet(DataConfig)
    for elem in iterator:
        # lab, comp, part, x, y = elem
        # print(lab)
        # pc = PointCloud()
        # pc.points = Vector3dVector(part)
        # o3d.visualization.draw_geometries([pc])
        #
        # pc = PointCloud()
        # pc.points = Vector3dVector(comp)
        # colors = []
        # t = 0
        # f = 0
        # for point in y:
        #     if point == 1:
        #         colors.append(np.array([0, 1, 0]))
        #         t += 1
        #     else:
        #         colors.append(np.array([1, 0, 0]))
        #         f += 1
        # # colors = np.stack(colors)
        # # colors = Vector3dVector(colors)
        # # pc.colors = colors
        # o3d.visualization.draw_geometries([pc], window_name="Green "+str(t) + ", red: " + str(f))
        pass
