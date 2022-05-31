from pathlib import Path
from random import uniform

import numpy as np
import torch
import torch.utils.data as data
import open3d as o3d
from open3d import visualization
try:
    from open3d.cuda.pybind import camera
except ImportError:
    from open3d.cpu.pybind import camera
from torch.utils.data import DataLoader

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
from utils.misc import sample_point_cloud
from scipy.spatial.transform import Rotation as R


class ShapeNet(data.Dataset):
    def __init__(self, config, mode="easy/train"):
        self.mode = mode
        self.data_root = Path(config.dataset_path)
        #  Backbone Input
        self.partial_points = config.partial_points
        # Implicit function input
        self.noise_rate = config.noise_rate
        self.implicit_input_dimension = config.implicit_input_dimension
        self.tolerance = config.tolerance
        self.dist = config.dist

        with (self.data_root / 'splits' / f'{self.mode}.txt').open('r') as file:
            lines = file.readlines()

        self.samples = lines
        self.n_samples = len(lines)

        with (self.data_root / 'splits' / mode.split('/')[0] / 'classes.txt').open('r') as file:
            self.labels_map = {l.split()[1]: l.split()[0] for l in file.readlines()}

        # Augmentation parameters
        self.camera_dist = {'low': 400, 'high': 1500}

        self.noise = config.noise  # Noise in the partial pc [the configuration one is for the sdf input]

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y

        # Find the mesh
        dir_path = self.data_root / 'data' / self.samples[idx].strip()
        label = int(self.labels_map[dir_path.parent.name])

        complete_path = dir_path / 'models/model_normalized.obj'

        # Load and randomly rotate the mesh
        mesh = o3d.io.read_triangle_mesh(str(complete_path), False)

        while True:
            rotation = R.random().as_matrix()
            mesh = mesh.rotate(rotation)

            # Define camera transformation and intrinsics
            #  (Camera is in the origin facing negative z, shifting it of z=1 puts it in front of the object)
            dist = uniform(**self.camera_dist)

            camera_parameters = camera.PinholeCameraParameters()
            camera_parameters.extrinsic = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, dist],
                                                    [0, 0, 0, 1]])

            intrinsics = {'fx': 1000, 'fy': 1000, 'cx': 959.5, 'cy': 539.5, 'width': 1920, 'height': 1080}
            camera_parameters.intrinsic.set_intrinsics(**intrinsics)

            # Move the view and take a depth image
            viewer = visualization.Visualizer()
            viewer.create_window(width=intrinsics['width'], height=intrinsics['height'], visible=False)
            viewer.clear_geometries()
            viewer.add_geometry(mesh)

            control = viewer.get_view_control()
            control.convert_from_pinhole_camera_parameters(camera_parameters)

            depth = viewer.capture_depth_float_buffer(True)

            viewer.remove_geometry(mesh)
            viewer.destroy_window()
            del control

            aux = depth[depth == 0.0]  # mesh too big or too small
            if np.all(aux) or not np.any(aux):
                continue

            depth = np.array(depth)
            if self.noise:  # from [paper link]: model of realsense d435 noise
                sigma = 0.001063 + 0.0007278 * (depth * 0.001) + 0.003949 * ((depth * 0.001) ** 2)
                depth += (np.random.normal(0, 1, depth.shape) * (depth != 0) * sigma * 1000)

            # Generate the partial point cloud
            depth_image = o3d.geometry.Image(depth)
            partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image,
                                                                          camera_parameters.intrinsic,
                                                                          camera_parameters.extrinsic)
            old_partial_pcd = np.array(partial_pcd.points)

            # Normalize the partial point cloud (all we could do at test time)
            mean = np.mean(old_partial_pcd, axis=0)
            partial_pcd = old_partial_pcd - mean
            var = np.sqrt(np.max(np.sum(partial_pcd ** 2, axis=1)))

            partial_pcd = partial_pcd / (var * 2)

            # Move the mesh so that it matches the partial point cloud position
            # (the [0, 0, 1] is to compensate for the fact that the partial pc is in the camera frame)
            mesh.translate(-mean)  #  + [0, 0, dist] (?)
            mesh.scale(1 / (var * 2), center=[0, 0, 0])

            # Make sure that the mesh normalized with the partial normalization is in the input space
            aux = np.array(mesh.vertices)
            t1 = np.all(np.min(aux, axis=0) > -0.5)
            t2 = np.all(np.max(aux, axis=0) < 0.5)

            if not (t1 and t2):
                continue

            break

        # Sample labeled point on the mesh
        samples, occupancy = sample_point_cloud(mesh,
                                                n_points=self.implicit_input_dimension,
                                                dist=self.dist,
                                                noise_rate=self.noise_rate,
                                                tolerance=self.tolerance)

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
            diff = self.partial_points - partial_pcd.shape[0]
            idx = np.random.choice(partial_pcd.shape[0], diff, replace=False)
            partial_pcd = torch.cat((partial_pcd, partial_pcd[idx]))

        samples = torch.tensor(samples).float()
        occupancy = torch.tensor(occupancy, dtype=torch.float)

        return label, partial_pcd, [np.array(mesh.vertices), np.array(mesh.triangles)], samples, occupancy

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
