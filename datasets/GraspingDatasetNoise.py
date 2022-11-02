import json
from pathlib import Path

import numpy as np
import torch as tc
import tqdm
from open3d import visualization

try:
    from open3d.cpu.pybind.geometry import TriangleMesh, PointCloud
    from open3d.cpu.pybind.io import read_point_cloud
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cpu.pybind import camera
except ImportError:
    from open3d.cuda.pybind.geometry import TriangleMesh, PointCloud
    from open3d.cuda.pybind.io import read_point_cloud
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cuda.pybind import camera
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import open3d as o3d


# class GraspingDataset(Dataset):
#     def __init__(self, root, json_file_path, subset, length=-1):
#         """
#         Args:
#         """
#         self.root = Path('./data/valid_set')
#
#
#     def __len__(self):
#         return 3200
#
#     def __getitem__(self, idx):
#         partial = np.load(self.root / f'partial_{idx:04}.npy')
#         gt = np.load(self.root / f'gt_{idx:04}.npy')
#
#         return partial.astype(np.float32), gt.astype(np.float32)


class GraspingDataset(Dataset):
    def __init__(self, root, json_file_path, subset, length=-1):
        """
        Args:
        """
        self.subset = subset
        self.file_path = json_file_path
        self.root = Path(root)
        with open(json_file_path, "r") as stream:
            self.data = json.load(stream)

        if length == -1:
            self.length = len(self.data[subset])
        else:
            self.length = length
        np.random.shuffle(self.data[subset])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            partial_path = self.data[self.subset][idx][0]
            ground_truth_path = self.data[self.subset][idx][1]

            # if not 'ycb' in ground_truth_path:
            #     return None

            vertices = np.load((self.root / ground_truth_path / 'vertices.npy').as_posix())
            triangles = np.load((self.root / ground_truth_path / 'triangles.npy').as_posix())

            p3, p2, p1 = Path(partial_path).parts[-1].split('_')[1:4]
            y1_pose = np.load(((self.root / partial_path).parent / f'_0_0_{p1}_model_pose.npy').as_posix())
            x2_pose = np.load(((self.root / partial_path).parent / f'_0_{p2}_0_model_pose.npy').as_posix())
            y3_pose = np.load(((self.root / partial_path).parent / f'_{p3}_0_0_model_pose.npy').as_posix())

            vertices = match_mesh_to_partial(np.array(vertices), [y1_pose, x2_pose, y3_pose])
            mesh = TriangleMesh(vertices=Vector3dVector(vertices), triangles=Vector3iVector(triangles))

            # Fix the mesh diameter to 0.25. While grasp_database meshes are already normalized to
            #  that value, ycb diameter is variable and during the depth rendering smaller meshes
            #  might cause empty depth images
            complete = np.array(mesh.sample_points_uniformly(8192 * 2).points)
            mesh = mesh.translate(get_bbox_center(complete)).scale(0.25 / get_diameter(complete), center=[0, 0, 0])

            # The partial point cloud is extracted with a camera and the reference frame is translated
            #  back to the object. If the depth was empty we move the camera closer. If it doesn't work we
            #  sample a new object
            partial = mesh_to_partial(mesh)
            if partial is None:
                partial = mesh_to_partial(mesh, dist=0.4)
            if partial is None:
                idx = np.random.randint(0, self.length)
                print('Resampling element')
                continue
            break

        complete = np.array(mesh.sample_points_uniformly(8192 * 2).points)

        choice = np.random.permutation(partial.shape[0])
        partial = partial[choice[:2048]]

        if partial.shape[0] < 2048:
            zeros = np.zeros((2048 - partial.shape[0], 3))
            partial = np.concatenate([partial, zeros])

        center = get_bbox_center(partial)
        diameter = get_diameter(partial - center)
        offset = np.array([0, 0, 0.5 - ((partial - center) / diameter * 0.7)[
            np.argmax(((partial - center) / diameter * 0.7)[..., 2])][..., 2]])

        complete, partial = ((complete - center) / diameter * 0.7) - offset, (
                    (partial - center) / diameter * 0.7) - offset

        return partial.astype(np.float32), complete.astype(np.float32)


def mesh_to_partial(mesh, dist=-1):
    if dist == -1:
        dist = np.random.randint(400, 1000) / 1000.

    t = np.block([[Rotation.from_euler('y', 0, degrees=True).as_matrix(), np.array([[0, 0, dist]]).T],
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
        return None

    sigma = 0.001063 + 0.0007278 * (depth) + 0.003949 * ((depth) ** 2)
    depth += (np.random.normal(0, 1, depth.shape) * (depth != 0) * sigma)

    # Generate the partial point cloud
    depth_image = o3d.geometry.Image(depth)
    partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image,
                                                                  camera_parameters.intrinsic,
                                                                  camera_parameters.extrinsic)
    return np.array(partial_pcd.points)


def get_bbox_center(pc):
    center = pc.min(0) + (pc.max(0) - pc.min(0)) / 2.0
    return center


def get_diameter(pc):
    diameter = np.max(pc.max(0) - pc.min(0))
    return diameter


def correct_pose(pose):
    pre = Rotation.from_euler('yz', [90, 180], degrees=True).as_matrix()
    post = Rotation.from_euler('zyz', [180, 180, 90], degrees=True).as_matrix()

    t = np.eye(4)
    t[:3, :3] = np.dot(post, np.dot(pose[:3, :3].T, pre))

    return t


def get_x_rot(pose):
    t = correct_pose(pose)

    x, y, z = Rotation.from_matrix(t[:3, :3].T).as_euler('xyz', degrees=True)

    rot_x = np.eye(4)
    rot_x[:3, :3] = Rotation.from_euler('x', x, degrees=True).as_matrix() @ Rotation.from_euler('x', 180,
                                                                                                degrees=True).as_matrix()

    return rot_x


def get_y_rot(pose):
    t = correct_pose(pose)

    x, y, z = Rotation.from_matrix(t[:3, :3].T).as_euler('xyz', degrees=True)
    if z < 0:
        t[:3, :3] = Rotation.from_euler('zxy', [-z, -x, y], degrees=True).as_matrix()
    else:
        t[:3, :3] = Rotation.from_euler('zxy', [-z, x, -y], degrees=True).as_matrix()

    theta_y = Rotation.from_matrix(t[:3, :3]).as_euler('zxy', degrees=True)[2]
    rot_y = np.eye(4)
    rot_y[:3, :3] = Rotation.from_euler('y', theta_y, degrees=True).as_matrix()

    return rot_y


def match_mesh_to_partial(vertices, pose):
    y1_pose, x2_pose, y3_pose = pose

    y_rot = get_y_rot(y1_pose)
    x2_rot = get_x_rot(x2_pose)
    y3_rot = get_y_rot(y3_pose)

    base_rot = np.eye(4)
    base_rot[:3, :3] = Rotation.from_euler('xyz', [180, 0, -90], degrees=True).as_matrix()

    t = y_rot @ x2_rot @ y3_rot @ base_rot

    pc = np.ones((np.size(vertices, 0), 4))
    pc[:, 0:3] = vertices

    pc = pc.T
    pc = t @ pc
    pc = pc.T[..., :3]

    return pc


if __name__ == '__main__':
    from open3d.visualization import draw_geometries

    root = 'data/MCD'
    split = 'data/MCD/build_datasets/train_test_dataset.json'

    training_set = GraspingDataset(root, split, subset='train_models_train_views', length=3200)
    Path('./valid_set').mkdir()
    for i, data in enumerate(tqdm.tqdm(training_set)):
        x, y = data
        np.save(f'partial_{i:04}', x)
        np.save(f'gt_{i:04}', y)



