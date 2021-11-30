from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
from open3d.visualization import draw_geometries
from torch import cdist
from utils.misc import create_3d_grid
import open3d as o3d
from configs.server_config import ModelConfig, DataConfig
from datasets.ShapeNetPOVRemoval import BoxNet
from main import HyperNetwork
import torch
import numpy as np
from genpose_sim import FromPartialToPose

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))

device = "cuda"


if __name__ == '__main__':

    res = 0.01

    model = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
    model = model.to(device)
    model.eval()

    generator = FromPartialToPose(model, res)

    valid_set = BoxNet(DataConfig, 10000)

    for sample in valid_set:

        label, partial, mesh_raw, x, y = sample

        verts, tris = mesh_raw

        mesh = o3d.geometry.TriangleMesh(Vector3dVector(verts), Vector3iVector(tris))
        o3d.visualization.draw_geometries([mesh])

        part = np.array(partial)

        complete = generator.reconstruct_point_cloud(part)
        p = generator.find_poses(complete, mult_res=2, n_points=100, iterations=1000)
        coords = generator.orient_poses(p)
        o3d.visualization.draw_geometries(coords + [complete])
