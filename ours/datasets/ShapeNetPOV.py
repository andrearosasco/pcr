import random
from pathlib import Path
from utils.misc import fps
import numpy as np
import torch
import torch.utils.data as data
import open3d as o3d
from numpy import cos, sin
from utils.misc import sample_point_cloud


class ShapeNet(data.Dataset):
    def __init__(self, config, mode="train"):
        self.mode = mode
        #  Backbone Input
        self.data_root = Path(config.dataset_path)
        self.partial_points = config.partial_points
        self.multiplier_complete_sampling = config.multiplier_complete_sampling

        # Implicit function input
        self.noise_rate = config.noise_rate
        self.percentage_sampled = config.percentage_sampled

        with (self.data_root / f'{self.mode}.txt').open('r') as file:
            lines = file.readlines()

        self.samples = lines
        self.n_samples = len(lines)

        with (self.data_root / 'classes.txt').open('r') as file:
            self.labels_map = {l.split()[1]: l.split()[0] for l in file.readlines()}

    @staticmethod
    def pc_norm(pcs):
        """ pc: NxC, return NxC """
        centroid = np.mean(pcs, axis=0)
        pcs = pcs - centroid
        m = np.max(np.sqrt(np.sum(pcs ** 2, axis=1)))
        pcs = pcs / m
        return pcs

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y
        print(self.data_root / self.samples[idx].strip())
        # Get label
        dir_path = self.data_root / self.samples[idx].strip()
        label = self.labels_map[dir_path.parent.name]

        # Extract point cloud from mesh
        tm = o3d.io.read_triangle_mesh(str(dir_path / 'models/model_normalized.obj'), True)  # ERRORE QUA!!!
        complete_pcd = tm.sample_points_uniformly(self.partial_points * self.multiplier_complete_sampling)

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
        partial_pcd = torch.FloatTensor(np.array(partial_pcd.points))

        # Set partial_pcd such that it has the same size of the others
        if partial_pcd.shape[0] < self.partial_points:
            diff = self.partial_points - partial_pcd.shape[0]
            partial_pcd = torch.cat((partial_pcd, torch.zeros(diff, 3)))
            print("[ShapeNetPOV] WARNING: padding incomplete point cloud")
        else:
            partial_pcd = fps(partial_pcd.unsqueeze(0), self.partial_points).squeeze()

        if self.mode == "valid":
            mesh_path = str(self.data_root / self.samples[idx].strip() / 'models/model_normalized.obj')
            return label, mesh_path, partial_pcd

        complete_pcd = np.array(complete_pcd.points)
        complete_pcd = torch.FloatTensor(complete_pcd)

        imp_x, imp_y = sample_point_cloud(tm,
                                          self.noise_rate,
                                          self.percentage_sampled)
        imp_x, imp_y = torch.tensor(imp_x).float(), torch.tensor(imp_y).bool().float().bool().float()  # TODO oh god..

        # Set partial_pcd such that it has the same size of the others
        if partial_pcd.shape[0] < self.partial_points:
            diff = self.partial_points - partial_pcd.shape[0]
            partial_pcd = torch.cat((partial_pcd, torch.zeros(diff, 3)))
            print(f"[ShapeNetPOV] WARNING: padding incomplete point cloud with {diff} points")
        else:
            partial_pcd = fps(partial_pcd.unsqueeze(0), self.partial_points).squeeze()

        if self.mode == "valid":
            mesh_path = str(self.data_root / self.samples[idx].strip() / 'models/model_normalized.obj')
            return label, mesh_path, partial_pcd

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