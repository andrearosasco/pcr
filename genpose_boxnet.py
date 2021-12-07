import time
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
from open3d.cpu.pybind.visualization import Visualizer
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

# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))

device = "cuda"


if __name__ == '__main__':

    res = 0.01

    model = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
    model = model.to(device)
    model.eval()

    generator = FromPartialToPose(model, res)

    valid_set = BoxNet(DataConfig, 10000)

    # Set up visualizer
    vis = Visualizer()
    vis.create_window()
    complete_pc = PointCloud()
    complete_pc.points = Vector3dVector(np.random.randn(2348, 3))
    partial_pc = PointCloud()
    partial_pc.points = Vector3dVector(np.random.randn(2024, 3))

    best1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    best2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    vis.add_geometry(best1)
    vis.add_geometry(best2)

    coords = []
    prevs = []
    for _ in range(6):
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coords.append(coord)
        prevs.append(np.array([0, 0, 1]))
        vis.add_geometry(coord)
    vis.add_geometry(complete_pc)
    vis.add_geometry(partial_pc)

    for sample in valid_set:
        break

    while True:

        # for coord in coords:
        #     vis.remove_geometry(coord)

        label, partial_points, mesh_raw, x, y = sample

        verts, tris = mesh_raw

        partial_points = np.array(partial_points)  # From list to array
        partial_pc_aux = PointCloud()
        partial_pc_aux.points = Vector3dVector(partial_points)

        # Reconstruct partial point cloud
        start = time.time()
        complete_pc_aux = generator.reconstruct_point_cloud(partial_points)
        print("Reconstruct: {}".format(time.time() - start))

        # Find poses
        p = generator.find_poses(complete_pc_aux, mult_res=1.5, n_points=100, iterations=1000, debug=False)

        # Orient poses
        # TODO START EXPERIMENT
        poses = p
        news = []
        highest_value = 0
        highest_id = -1
        lowest_value = 0
        lowest_id = -1
        i = 0
        for c, normal, prev, coord in zip(np.array(poses.points), np.array(poses.normals), prevs, coords):
            coord.translate(c, relative=False)
            normal = normal/np.linalg.norm(normal)
            R = FromPartialToPose.create_rotation_matrix(prev, normal)
            coord.rotate(R, center=c)
            vis.update_geometry(coord)
            news.append(R @ prev)

            if normal[0] > highest_value:
                highest_value = normal[0]
                highest_id = i
            if normal[0] < lowest_value:
                lowest_value = normal[0]
                lowest_id = i

            i += 1
        prevs = news
        # TODO END EXPERIMENT
        # coords = generator.orient_poses(p)  # TODO BEFORE

        # Update coords
        # for coord in coords:
        #     vis.add_geometry(coord)

        # Update partial point cloud in visualizer
        partial_pc.clear()
        partial_pc += partial_pc_aux
        colors = np.array([0, 255, 0])[None, ...].repeat(len(partial_pc.points), axis=0)
        partial_pc.colors = Vector3dVector(colors)
        vis.update_geometry(partial_pc)

        # Update complete point cloud in visualizer
        complete_pc.clear()
        complete_pc += complete_pc_aux
        colors = np.array([255, 0, 0])[None, ...].repeat(len(complete_pc.points), axis=0)
        complete_pc.colors = Vector3dVector(colors)
        vis.update_geometry(complete_pc)

        # Update best points
        best1.translate(np.array(poses.points)[highest_id], relative=False)
        best2.translate(np.array(poses.points)[lowest_id], relative=False)
        vis.update_geometry(best1)
        vis.update_geometry(best2)

        # Update visualizer
        vis.poll_events()
        vis.update_renderer()

        # o3d.visualization.draw_geometries(coords + [complete])

# GOOD: 1, 10, 100  BAD
# GOOD: 1, 10, 1000  EXCELLENT
# GOOD: 1, 100, 1000 CAN BE BETTER
# GOOD: 1, 1000, 1000
# GOOD: 2, 100, 100

# mesh = o3d.geometry.TriangleMesh(Vector3dVector(verts), Vector3iVector(tris))
# o3d.visualization.draw_geometries([mesh])

# for i in [0.1, 0.5, 1, 2]:
#     for k in [10, 100, 1000]:
#         for j in [10, 100, 1000]:
# print("i: {}, k:{}, j:{}".format(i, k, j))
