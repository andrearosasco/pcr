import random
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
import open3d as o3d
from configs.local_config import TrainConfig
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
from numpy import cos, sin
from utils.misc import sample_point_cloud


class ShapeNet(data.Dataset):
    def __init__(self, config, mode="train", overfit_mode=False):
        self.mode = mode
        self.overfit_mode = overfit_mode
        #  Backbone Input
        self.data_root = Path(config.dataset_path)
        self.partial_points = config.partial_points
        self.multiplier_complete_sampling = config.multiplier_complete_sampling

        # Implicit function input
        self.noise_rate = config.noise_rate
        self.percentage_sampled = config.percentage_sampled
        self.implicit_input_dimension = config.implicit_input_dimension

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

    # TODO refactor, it's a mess.
    # TODO different behavior for test e valid in the same get_item. Maybe 2 get item and an if
    # TODO maybe return just the mesh and the partial pcd in any case but then impx computation is not parallelized

    # TODO uniform sampling to avoid padding

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y
        padding_length = 0

        # Extract point cloud from mesh
        dir_path = self.data_root / self.samples[idx].strip()
        label = int(self.labels_map[dir_path.parent.name])
        complete_path = str(dir_path / 'models/model_normalized.obj')

        if self.overfit_mode:
            complete_path = TrainConfig.overfit_sample

        tm = o3d.io.read_triangle_mesh(complete_path, False)
        complete_pcd = tm.sample_points_uniformly(self.partial_points * self.multiplier_complete_sampling)

        # Get random position of camera
        sph_radius = 1
        y = random.uniform(-sph_radius, sph_radius)
        theta = random.uniform(0, 2 * np.pi)
        x = np.sqrt(sph_radius ** 2 - y ** 2) * cos(theta)
        z = np.sqrt(sph_radius ** 2 - y ** 2) * sin(theta)
        camera = [x, y, z]

        # Center to be in the middle
        # points = np.array(complete_pcd.points)
        # center = [max((points[:, 0] + min(points[:, 0]))/2),
        #           max((points[:, 1] + min(points[:, 1]))/2),
        #           max((points[:, 2] + min(points[:, 2]))/2)]
        # center = np.array(center)[None, ...].repeat(len(points), axis=0)
        # complete_pcd.points = Vector3dVector(points - center)

        # Remove hidden points
        _, pt_map = complete_pcd.hidden_point_removal(camera, 500)  # radius * 4
        partial_pcd = complete_pcd.select_by_index(pt_map)

        partial_pcd = torch.FloatTensor(np.array(partial_pcd.points))
        # Set partial_pcd such that it has the same size of the others
        if partial_pcd.shape[0] < self.partial_points:
            diff = self.partial_points - partial_pcd.shape[0]
            partial_pcd = torch.cat((partial_pcd, torch.zeros(diff, 3)))
            padding_length = diff

        else:
            perm = torch.randperm(partial_pcd.size(0))
            idx = perm[:self.partial_points]
            partial_pcd = partial_pcd[idx]

        if self.mode in ['valid', 'test']:
            if self.overfit_mode:
                mesh_path = TrainConfig.overfit_sample
            else:
                mesh_path = str(self.data_root / self.samples[idx].strip() / 'models/model_normalized.obj')
            return label, partial_pcd, mesh_path,

        complete_pcd = np.array(complete_pcd.points)
        complete_pcd = torch.FloatTensor(complete_pcd)

        imp_x, imp_y = sample_point_cloud(tm,
                                          self.noise_rate,
                                          self.percentage_sampled,
                                          total=self.implicit_input_dimension,
                                          mode="unsigned")
        imp_x, imp_y = torch.tensor(imp_x).float(), torch.tensor(imp_y).bool().float().bool().float()  # TODO oh god..

        return label, partial_pcd, complete_pcd, imp_x, imp_y, padding_length

    def __len__(self):
        return int(self.n_samples / (100 if self.overfit_mode else 1))


if __name__ == "__main__":
    from ours.configs.local_config import DataConfig
    from tqdm import tqdm
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector

    a = DataConfig()
    a.dataset_path = Path("..", "..", "data", "ShapeNetCore.v2")
    iterator = ShapeNet(a)
    for elem in tqdm(iterator):
        lab, part, comp, x, y, pad = elem
        pass

        pc = PointCloud()
        pc.points = Vector3dVector(comp)
        o3d.visualization.draw_geometries([pc], window_name="Complete")
        #
        pc = PointCloud()
        pc.points = Vector3dVector(part)
        o3d.visualization.draw_geometries([pc], window_name="Partial")

        # print(lab)
        #
        # points = []
        # for _ in range(1000):
        #     points.append(np.array([1, random.uniform(-1, 1), random.uniform(-1, 1)]))
        #     points.append(np.array([-1, random.uniform(-1, 1), random.uniform(-1, 1)]))
        #     points.append(np.array([random.uniform(-1, 1), 1, random.uniform(-1, 1)]))
        #     points.append(np.array([random.uniform(-1, 1), -1, random.uniform(-1, 1)]))
        #     points.append(np.array([random.uniform(-1, 1), random.uniform(-1, 1), 1]))
        #     points.append(np.array([random.uniform(-1, 1), random.uniform(-1, 1), -1]))
        #
        # points = np.stack(points)
        # points = np.concatenate((points, comp))

        # pc = PointCloud()
        # pc.points = Vector3dVector(x)
        # colors = []
        # for i in y:
        #     if i == 0.:
        #         colors.append(np.array([1, 0, 0]))
        #     if i == 1.:
        #         colors.append(np.array([0, 1, 0]))
        # colors = np.stack(colors)
        # colors = Vector3dVector(colors)
        # pc.colors = colors
        # o3d.visualization.draw_geometries([pc])
