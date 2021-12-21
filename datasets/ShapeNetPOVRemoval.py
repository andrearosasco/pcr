from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
import open3d as o3d
from torch.utils.data import DataLoader

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
from utils.misc import sample_point_cloud
from scipy.spatial.transform import Rotation as R


class ShapeNet(data.Dataset):
    def __init__(self, config, mode="easy/train", overfit_mode=False):
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

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y

        # Find the mesh
        dir_path = self.data_root / self.samples[idx].strip()
        label = int(self.labels_map[dir_path.parent.name])
        complete_path = dir_path / 'models/model_normalized.obj'

        # Load and randomly rotate the mesh
        mesh = o3d.io.read_triangle_mesh(str(complete_path), False)

        while True:
            rotation = R.random().as_matrix()
            mesh = mesh.rotate(rotation)

            # Define camera transformation and intrinsics
            #  (Camera is in the origin facing negative z, shifting it of z=1 puts it in front of the object)
            dist = 1.5

            complete_pcd = mesh.sample_points_uniformly(self.partial_points * self.multiplier_complete_sampling)
            _, pt_map = complete_pcd.hidden_point_removal([0, 0, dist], 1000)  # radius * 4
            partial_pcd = complete_pcd.select_by_index(pt_map)
            #
            # partial_pcd.paint_uniform_color([0, 0, 1])
            # complete_pcd.paint_uniform_color([1, 0, 0])
            # draw_geometries([partial_pcd, complete_pcd])

            if len(np.array(partial_pcd.points)) != 0:
                break

        # Normalize the partial point cloud (all we could do at test time)
        partial_pcd = np.array(partial_pcd.points)
        mean = np.mean(np.array(partial_pcd), axis=0)
        partial_pcd = np.array(partial_pcd) - mean
        var = np.sqrt(np.max(np.sum(partial_pcd ** 2, axis=1)))

        partial_pcd = partial_pcd / (var * 2)

        # Move the mesh so that it matches the partial point cloud position
        # (the [0, 0, 1] is to compensate for the fact that the partial pc is in the camera frame)
        mesh.translate(-mean)
        mesh.scale(1 / (var * 2), center=[0, 0, 0])

        # Sample labeled point on the mesh
        samples, occupancy = sample_point_cloud(mesh,
                                                self.noise_rate,
                                                self.percentage_sampled,
                                                total=self.implicit_input_dimension,
                                                mode="unsigned")

        # Next lines bring the shape a face of the cube so that there's more space to
        # complete it. But is it okay for the input to be shifted toward -0.5 and not
        # centered on the origin?
        #
        # normalized[..., 2] = normalized[..., 2] + (-0.5 - min(normalized[..., 2]))

        partial_pcd = torch.FloatTensor(partial_pcd)

        # Set partial_pcd such that it has the same size of the others
        if partial_pcd.shape[0] > self.partial_points:
            perm = torch.randperm(partial_pcd.size(0))
            ids = perm[:self.partial_points]
            partial_pcd = partial_pcd[ids]
        else:
            print(f'Warning: had to pad the partial pcd {complete_path} - points {partial_pcd.shape[0]} added {self.partial_points - partial_pcd.shape[0]}')
            diff = self.partial_points - partial_pcd.shape[0]
            partial_pcd = torch.cat((partial_pcd, torch.zeros(diff, 3)))

        samples = torch.tensor(samples).float()
        occupancy = torch.tensor(occupancy, dtype=torch.float) / 255

        return label, partial_pcd, [str(complete_path), rotation, mean, var], samples, occupancy

    def __len__(self):
        return int(self.n_samples)


if __name__ == "__main__":
    from ours.configs.local_config import DataConfig
    from tqdm import tqdm
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector

    a = DataConfig()
    a.dataset_path = Path("..", "data", "ShapeNetCore.v2")
    iterator = ShapeNet(a)
    loader = DataLoader(iterator, num_workers=0, shuffle=False, batch_size=1)
    for elem in tqdm(loader):
        lab, part, comp, x, y = elem

        # pc = PointCloud()
        # pc.points = Vector3dVector(comp)
        # o3d.visualization.draw_geometries([pc], window_name="Complete")
        # #
        # pc = PointCloud()
        # pc.points = Vector3dVector(part)
        # o3d.visualization.draw_geometries([pc], window_name="Partial")

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
