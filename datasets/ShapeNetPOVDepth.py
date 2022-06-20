import gc
from pathlib import Path
from numpy.random import uniform
import numpy as np
import torch
import torch.utils.data as data
import open3d as o3d
from open3d import visualization
try:
    from open3d.cuda.pybind import camera
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
except ImportError:
    from open3d.cpu.pybind import camera
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    # from open3d.open3d import camera
    # from open3d.open3d.utility import Vector3dVector, Vector3iVector
from torch.utils.data import DataLoader

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
from utils.misc import sample_point_cloud, read_mesh_debug, read_mesh
from scipy.spatial.transform import Rotation as R, Rotation

class ShapeNet(data.Dataset):
    def __init__(self, config, mode):
        self.mode = mode
        self.data_root = Path(config.Data.dataset_path)
        #  Backbone Input
        self.partial_points = config.Data.partial_points
        # Implicit function input
        self.implicit_input_dimension = config.Data.implicit_input_dimension
        # Augmentation parameters
        self.tolerance = config.Data.tolerance
        self.dist = config.Data.dist
        self.noise_rate = config.Data.noise_rate
        self.offset = config.Data.offset

        self.seed = config.General.seed

        with (self.data_root / f'{self.mode}.txt').open('r') as file:
            lines = file.readlines()
        # lines = []
        # aux = []
        # with open(f'../../Desktop/debug6.txt', 'r+', encoding='utf8') as f:
        #     for l in f.readlines():
        #         if l != '\n' and l[:5] != 'Epoch':
        #             lines.append(l.strip()[26:-28])
        #             aux.append(l)

        self.samples = lines
        self.n_samples = len(lines)

        with (self.data_root / 'classes.txt').open('r') as file:
            self.labels_map = {l.split()[1]: l.split()[0] for l in file.readlines()}

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y

        while True:
            dir_path = self.data_root / 'data' / self.samples[idx].strip()
            label = int(self.labels_map[dir_path.parent.name])

            v = np.load(str(dir_path / 'models/model_vertices.npy'))
            t = np.load(str(dir_path / 'models/model_triangles.npy'))
            mesh = o3d.geometry.TriangleMesh(Vector3dVector(v),
                                             Vector3iVector(t))

            # mesh = read_mesh_debug(dir_path)

            rotation = R.random().as_matrix()
            mesh = mesh.rotate(rotation)

            # print(f'{complete_path} stuck {i} {j}')

            # Define camera transformation and intrinsics
            #  (Camera is in the origin facing negative z, shifting it of z=1 puts it in front of the object)
            dist = 1
            t = np.block([[Rotation.from_euler('y', 180, degrees=True).as_matrix(), np.array([[0, 0, dist]]).T],
                          [np.eye(4)[3]]])

            camera_parameters = camera.PinholeCameraParameters()
            camera_parameters.extrinsic = t

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
            # print(f'{idx}: Extracted depth image', file=Path(f'./debug/{os.getpid()}.txt').open('a+'))

            viewer.remove_geometry(mesh)
            viewer.destroy_window()
            del control
            del viewer

            depth = np.array(depth)

            aux = depth == 0.0  # mesh too big or too small
            if np.all(aux) or not np.any(aux):
                idx = np.random.randint(low=0, high=len(self))
                continue

            # Generate the partial point cloud
            depth_image = o3d.geometry.Image(depth)
            partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image,
                                                                          camera_parameters.intrinsic,
                                                                          camera_parameters.extrinsic)
            # print(f'{idx}: Generated Point Cloud', file=Path(f'./debug/{os.getpid()}.txt').open('a+'))
            old_partial_pcd = np.array(partial_pcd.points)

            # Normalize the partial point cloud (all we could do at test time)
            complete = np.array(mesh.sample_points_uniformly(8192).points)
            mean = np.mean(complete, axis=0)
            partial_pcd = old_partial_pcd - mean
            var = np.sqrt(np.max(np.sum((complete - mean) ** 2, axis=1)))

            if var == 0:
                idx = np.random.randint(low=0, high=len(self))
                continue

            partial_pcd = partial_pcd / (var * 2)

            offset = np.zeros([3])
            if self.offset:
                offset = 0.5 - partial_pcd[np.argmax(partial_pcd[..., 2])][..., 2]
                offset = np.array([0, 0, offset])
                partial_pcd = partial_pcd + offset

            # Move the mesh so that it matches the partial point cloud position
            # (the [0, 0, 1] is to compensate for the fact that the partial pc is in the camera frame)
            mesh.translate(-mean)  #  + [0, 0, dist] (?)
            mesh.scale(1 / (var * 2), center=[0, 0, 0])

            if self.offset:
                mesh.translate(offset)

            # Make sure that the mesh normalized with the partial normalization is in the input space

            # aux = np.array(mesh.vertices)
            # t1 = np.all(np.min(aux, axis=0) > -0.5)
            # t2 = np.all(np.max(aux, axis=0) < 0.5)
            #
            # if not (t1 and t2):
            #     continue

            break

        # print(f'{idx}: End of while', file=Path(f'./debug/{os.getpid()}.txt').open('a+'))
        # Sample labeled points on the mesh
        samples, occupancy = sample_point_cloud(mesh,
                                                n_points=self.implicit_input_dimension,
                                                dist=self.dist,
                                                noise_rate=self.noise_rate,
                                                tolerance=self.tolerance,
                                                seed=self.seed)
        # print(f'{idx}: Implicit imput computed', file=Path(f'./debug/{os.getpid()}.txt').open('a+'))

        # Next lines bring the shape a face of the cube so that there's more space to
        # complete it. But is it okay for the input to be shifted toward -0.5 and not
        # centered on the origin?
        #
        # normalized[..., 2] = normalized[..., 2] + (-0.5 - min(normalized[..., 2]))

        partial_pcd = torch.FloatTensor(partial_pcd)

        # Set partial_pcd such that it has the same size of the others
        # If the points are less than (self.partial_points / 2) then diff is higher than partial_pcd.shape[0] and
        # replace True is needed.
        if partial_pcd.shape[0] > self.partial_points:
            perm = torch.randperm(partial_pcd.size(0))
            ids = perm[:self.partial_points]
            partial_pcd = partial_pcd[ids]
        else:
            diff = self.partial_points - partial_pcd.shape[0]
            idx = np.random.choice(partial_pcd.shape[0], diff, replace=True)
            partial_pcd = torch.cat((partial_pcd, partial_pcd[idx]))

        samples = torch.tensor(samples).float()
        occupancy = torch.tensor(occupancy, dtype=torch.float)

        # print(f'{idx}: Returning output', file=Path(f'./debug/{os.getpid()}.txt').open('a+'))
        # print('rotation', type(rotation))
        # print('mean', type(mean))
        # print('var', type(var))
        return label, partial_pcd, [str(dir_path), rotation, mean, var, offset], samples, occupancy

    def __len__(self):
        return int(self.n_samples)

if __name__ == "__main__":
    from configs import Config
    from tqdm import tqdm
    from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh

    iterator = ShapeNet(Config, mode='hard/train')
    loader = DataLoader(iterator, num_workers=8, shuffle=False, batch_size=8)
    for elem in tqdm(loader):
        lab, part, comp, x, y = elem
        print(lab)

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
