import json
from pathlib import Path

import numpy as np
import torch as tc
import tqdm

from utils.misc import create_cube

try:
    from open3d.cpu.pybind.geometry import TriangleMesh, PointCloud
    from open3d.cpu.pybind.io import read_point_cloud
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cpu.pybind.visualization import draw_geometries
except ImportError:
    from open3d.cuda.pybind.geometry import TriangleMesh, PointCloud
    from open3d.cuda.pybind.io import read_point_cloud
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

patch_size = 40


class GraspingDataset(Dataset):
    def __init__(self, root, json_file_path, subset='train_models_train_views'):
        """
        Args:
        """
        self.subset = subset
        self.file_path = json_file_path
        self.root = Path(root)
        with open(json_file_path, "r") as stream:
            self.data = json.load(stream)

        self.length = len(self.data[subset])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        partial_path = self.data[self.subset][idx][0]
        ground_truth_path = self.data[self.subset][idx][1]

        vertices = np.load((self.root / ground_truth_path / 'vertices.npy').as_posix())
        triangles = np.load((self.root / ground_truth_path / 'triangles.npy').as_posix())

        p3, p2, p1 = Path(partial_path).parts[-1].split('_')[1:4]
        y1_pose = np.load(((self.root / partial_path).parent / f'_0_0_{p1}_model_pose.npy').as_posix())
        x2_pose = np.load(((self.root / partial_path).parent / f'_0_{p2}_0_model_pose.npy').as_posix())
        y3_pose = np.load(((self.root / partial_path).parent / f'_{p3}_0_0_model_pose.npy').as_posix())


        vertices = match_mesh_to_partial(np.array(vertices), [y1_pose, x2_pose, y3_pose])
        mesh = TriangleMesh(vertices=Vector3dVector(vertices), triangles=Vector3iVector(triangles))
        complete = np.array(mesh.sample_points_uniformly(8192 * 2).points)

        partial = np.load(self.root / (partial_path + 'partial.npy'))
        # partial = np.array(read_point_cloud((self.root / (partial_path + 'pc.pcd')).as_posix()).points)

        partial = partial + np.array([0, 0, -1])

        choice = np.random.permutation(partial.shape[0])
        partial = partial[choice[:2048]]

        if partial.shape[0] < 2048:
            zeros = np.zeros((2048 - partial.shape[0], 3))
            partial = np.concatenate([partial, zeros])

        center = get_bbox_center(complete)
        # diameter = np.sqrt(np.max(np.sum((complete - center) ** 2, axis=1))) * 2
        diameter = get_diameter(complete - center)

        complete, partial = (complete - center) / diameter, (partial - center) / diameter

        return partial.astype(np.float32), complete.astype(np.float32)

    def get_item_name(self, string):
        string = string[:-6]
        string_list = string.split("/")
        idx = -1
        for i, j in enumerate(string_list):
            if j == "ycb" or j == "grasp_database":
                idx = i
        return string_list[idx + 1] + string_list[-1]


def pc_to_binvox_for_shape_completion(points,
                                      patch_size):
    """
    This function creates a binvox object from a pointcloud.  The voxel grid is slightly off center from the
    pointcloud bbox center so that the back of the grid has more room for the completion.
    :type points: numpy.ndarray
    :param points: nx3 numpy array representing a pointcloud
    :type patch_size: int
    :param patch_size: how many voxels along a single dimension of the voxel grid.
    Ex: patch_size=40 gives us a 40^3 voxel grid
    :rtype: binvox_rw.Voxels
    """

    if points.shape[1] != 3:
        raise Exception("Invalid pointcloud size, should be nx3, but is {}".format(points.shape))

    # how much of the voxel grid do we want our pointcloud to fill.
    # make this < 1 so that there is some padding on the edges
    PERCENT_PATCH_SIZE = (4.0 / 5.0)

    # Where should the center of the points be placed inside the voxel grid.
    # normally make PERCENT_Z < 0.5 so that the points are placed towards the front of the grid
    # this leaves more room for the shape completion to fill in the occluded back half of the occupancy grid.
    PERCENT_X = 0.5
    PERCENT_Y = 0.5
    PERCENT_Z = 0.45

    # get the center of the pointcloud in meters. Ex: center = np.array([0.2, 0.1, 2.0])
    center = get_bbox_center(points)

    # get the size of an individual voxel. Ex: voxel_resolution=0.01 meaning 1cm^3 voxel
    # PERCENT_PATCH_SIZE determines how much extra padding to leave on the sides
    voxel_resolution = get_voxel_resolution(points, PERCENT_PATCH_SIZE * patch_size)

    # this tuple is where we want to stick the center of the pointcloud in our voxel grid
    # Ex: (20, 20, 18) leaving some extra room in the back half.
    pc_center_in_voxel_grid = (patch_size * PERCENT_X, patch_size * PERCENT_Y, patch_size * PERCENT_Z)

    # create a voxel grid.
    vox_np = voxelize_points(
        points=points[:, 0:3],
        pc_bbox_center=center,
        voxel_resolution=voxel_resolution,
        num_voxels_per_dim=patch_size,
        pc_center_in_voxel_grid=pc_center_in_voxel_grid)

    # location in meters of the bottom corner of the voxel grid in world space
    offset = np.array(center) - np.array(pc_center_in_voxel_grid) * voxel_resolution

    # create a voxel grid object to contain the grid, shape, offset in the world, and grid resolution
    vox = binvox_rw.Voxels(vox_np, vox_np.shape, tuple(offset), voxel_resolution * patch_size, "xyz")
    return vox


def voxelize_points(points, pc_bbox_center, voxel_resolution, num_voxels_per_dim, pc_center_in_voxel_grid):
    """
    This function takes a pointcloud and produces a an occupancy map or voxel grid surrounding the points.
    :type points: numpy.ndarray
    :param points: an nx3 numpy array representing a pointcloud
    :type pc_bbox_center: numpy.ndarray
    :param pc_bbox_center: numpy.ndarray of shape (3,) representing the center of the bbox that contains points
    :type voxel_resolution: float
    :param voxel_resolution: float describing in meters the length of an individual voxel edge. i.e 0.01 would
    mean each voxel is 1cm^3
    :type num_voxels_per_dim: int
    :param num_voxels_per_dim: how many voxels along a dimension. normally 40, for a 40x40x40 voxel grid
    :type pc_center_in_voxel_grid: tuple
    :param pc_center_in_voxel_grid: (x,y,z) in voxel coords of where to place the center of the points in the voxel grid
    if using 40x40x40 voxel grid, then pc_center_in_voxel_grid = (20,20,20).  We often using something more
    like (20,20,18) when doing shape completion so there is more room in the back of the grid for the
    object to be completed.
    """

    # this is the voxel grid we are going to return
    voxel_grid = np.zeros((num_voxels_per_dim,
                           num_voxels_per_dim,
                           num_voxels_per_dim), dtype=np.bool)

    # take the points and convert them from meters to voxel space coords
    centered_scaled_points = np.floor(
        (points - np.array(pc_bbox_center) + np.array(
            pc_center_in_voxel_grid) * voxel_resolution) / voxel_resolution)

    # remove any points that are beyond the area that falls in our voxel grid
    mask = centered_scaled_points.max(axis=1) < num_voxels_per_dim
    centered_scaled_points = centered_scaled_points[mask]

    # if we don't have any more points that fall within our voxel grid
    # return an empty grid
    if centered_scaled_points.shape[0] == 0:
        return voxel_grid

    # remove any points that are outside of the region we are voxelizing
    # as they are to small.
    mask = centered_scaled_points.min(axis=1) > 0
    centered_scaled_points = centered_scaled_points[mask]

    # if we don't have any more points that fall within our voxel grid,
    # return an empty grid
    if centered_scaled_points.shape[0] == 0:
        return voxel_grid

    # treat our remaining points as ints, since we are already in voxel coordinate space.
    # this points shoule be things like (5, 6, 7) which represent indices in the voxel grid.
    csp_int = centered_scaled_points.astype(int)

    # create a mask from our set of points.
    mask = (csp_int[:, 0], csp_int[:, 1], csp_int[:, 2])

    # apply the mask to our voxel grid setting voxel that had points in them to be occupied
    voxel_grid[mask] = 1

    return voxel_grid


def get_voxel_resolution(pc, patch_size):
    min_x = pc[:, 0].min()
    min_y = pc[:, 1].min()
    min_z = pc[:, 2].min()
    max_x = pc[:, 0].max()
    max_y = pc[:, 1].max()
    max_z = pc[:, 2].max()

    max_dim = max((max_x - min_x),
                  (max_y - min_y),
                  (max_z - min_z))

    voxel_resolution = (1.0 * max_dim) / patch_size

    return voxel_resolution


def get_bbox_center(pc):
    center = pc.min(0) + (pc.max(0) - pc.min(0)) / 2.0
    return center


def get_diameter(pc):
    diameter = pc.max(0) - pc.min(0)
    return np.max(diameter)


class deterRandomSampler(Sampler):
    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        tc.manual_seed(self.seed)
        return iter(tc.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)


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
    root = 'data/MCD'
    split = 'data/MCD/build_datasets/train_test_dataset.json'

    training_set = MCDataset(root, split, subset='train_models_train_views')
    for data in tqdm.tqdm(training_set):
        x, y = data

        a = PointCloud(points=Vector3dVector(x))
        a.paint_uniform_color([1, 0, 0])

        b = PointCloud(points=Vector3dVector(y))
        b.paint_uniform_color([0, 1, 0])

        draw_geometries([a, b, create_cube()])