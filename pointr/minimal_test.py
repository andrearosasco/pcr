from torch import randn_like
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector
from open3d.open3d.visualization import draw_geometries
import open3d
from datasets.ShapeNet55Dataset import ShapeNet
from models.PoinTr import PoinTr
import torch
import numpy as np
from utils import misc
import time


def draw_point_cloud(x):
    x = x.cpu().squeeze()
    pc = PointCloud()
    pc.points = Vector3dVector(np.array(x))
    draw_geometries([pc])


crop_ratio = {
    'easy': 1/4,
    'median': 1/2,
    'hard': 3/4
}


# Load Dataset
class AttrDict(dict):
    def _init_(self, *args, **kwargs):
        super(AttrDict, self)._init_(*args, **kwargs)
        self._dict_ = self


n_points = 8192
dataset_name = "ShapeNet"

class data_config:
    def __init__(self):
        self.DATA_PATH = '../data/ShapeNet55-34/ShapeNet-55'
        self.NAME = 'ShapeNet'
        self.N_POINTS = 8192
        self.subset = 'test'
        self.PC_PATH = '../data/ShapeNet55-34/shapenet_pc'


c = data_config()
dataset = ShapeNet(c)

# Load model
class model_config:
    def __init__(self):
        self.NAME = 'PoinTr'
        self.knn_layer = 1
        self.num_pred = 6144
        self.num_query = 96
        self.trans_dim = 384

c = model_config()
model = PoinTr(c)

# Load checkpoint
state_dict = torch.load("pretrained/PoinTr_ShapeNet55.pth")  # dict of model info
base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
model.load_state_dict(base_ckpt)
model.cuda()
model.eval()


def sample_point_cloud(xyz, voxel_size=0.1, noise_rate=0.1, percentage_sampled=0.1):  # 1 2048 3
    # TODO try also with https://blender.stackexchange.com/questions/31693/how-to-find-if-a-point-is-inside-a-mesh
    # VOXEL
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    voxel_grid = open3d.open3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    # open3d.open3d.visualization.draw_geometries([voxel_grid])

    # ORIGINAL PC
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    # open3d.open3d.visualization.draw_geometries([pcd])

    # WITH GAUSSIAN NOISE
    z = randn_like(xyz) * noise_rate
    xyz = xyz + z
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    # open3d.open3d.visualization.draw_geometries([pcd])

    # WITH 10% UNIFORM RANDOM POINTS
    k = int(len(xyz) * percentage_sampled)
    random_points = torch.FloatTensor(k, 3).uniform_(-1, 1)
    pcd = open3d.geometry.PointCloud()
    xyz = torch.cat((xyz, random_points))
    pcd.points = open3d.utility.Vector3dVector(xyz)
    # open3d.open3d.visualization.draw_geometries([pcd])

    # WITH LABEL
    results = voxel_grid.check_if_included(open3d.utility.Vector3dVector(xyz))
    colors = np.zeros((len(xyz), 3))
    for i, value in enumerate(results):
        if value:
            colors[i] = np.array([0, 0, 1])
        else:
            colors[i] = np.array([0, 1, 0])
    pcd.colors = open3d.utility.Vector3dVector(colors)
    open3d.open3d.visualization.draw_geometries([pcd])

    t = 0
    f = 0
    for elem in results:
        if elem:
            t += 1
        else:
            f += 1
    print("Found ", t, " points inside the voxels and ", f, " points outside the voxel")

    return np.array(pcd.points), np.array(results)


with torch.no_grad():
    for idx, (taxonomy_ids, model_ids, data) in enumerate(dataset):
        taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
        model_id = model_ids[0]

        # TODO SUBSAMPLE GT CLOUD
        x, y = sample_point_cloud(data)
        # TODO END

        gt = data.cuda()
        choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]), torch.Tensor([-1, 1, 1]),
                  torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
                  torch.Tensor([-1, -1, -1])]
        num_crop = int(n_points * crop_ratio["easy"])
        for item in choice:
            partial, _ = misc.seprate_point_cloud(gt.unsqueeze(0), n_points, num_crop, fixed_points=item)
            partial = misc.fps(partial, 2048)

            start = time.time()
            ret = model(partial)
            end = time.time()

            # print(end-start)

            coarse_points = ret[0]
            dense_points = ret[1]

            # draw_point_cloud(gt)
            # draw_point_cloud(partial)
            # draw_point_cloud(coarse_points)
            # draw_point_cloud(dense_points)



