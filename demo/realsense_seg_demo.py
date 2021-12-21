import copy
import time
from pathlib import Path

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import Visualizer, draw_geometries
from sklearn.cluster import DBSCAN
# from cuml.cluster import DBSCAN
import open3d as o3d

from configs import ModelConfig
from utils.input import RealSense
from utils.misc import create_3d_grid

vis = Visualizer()
vis.create_window('Pose Estimation')
camera = RealSense(640, 480)

full_pcd = PointCloud()
render_setup = False
i = 0
device = 'cuda'

coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
tot = 0
while True:
    rgb, depth = camera.read()
    #
    # cv2.imshow('rgb', rgb)
    # cv2.imshow('depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))
    # cv2.waitKey(1)

    depth[depth > 2000] = 0
    pc = camera.pointcloud(depth)
    pcd = PointCloud()
    pcd.points = Vector3dVector(pc)
    whole_pc = copy.deepcopy(pcd)

    start = time.time()

    idx = np.random.choice(pc.shape[0], (int(pc.shape[0] * 0.05)), replace=False)
    pc = pc[idx]

    if pc.shape[0] == 0:
        continue

    clustering = DBSCAN(eps=0.1, min_samples=10).fit(pc)
    close = clustering.labels_[pc.argmax(axis=0)[2]]

    pc = pc[clustering.labels_ == close]

    partial = torch.FloatTensor(pc)  # Must be 1, 2024, 3

    # Normalize Point Cloud as training time
    partial = np.array(partial)
    mean = np.mean(np.array(partial), axis=0)
    partial = np.array(partial) - mean
    var = np.sqrt(np.max(np.sum(partial ** 2, axis=1)))
    partial = partial / (var * 2)

    # tot += (time.time() - start)

    if partial.shape[0] > 2500:
        continue

    part_pc = PointCloud()
    part_pc.points = Vector3dVector(partial)

    part_pc.paint_uniform_color([1, 0, 0])
    full_pcd.clear()
    whole_pc.paint_uniform_color([0, 1, 0])
    full_pcd += (part_pc + whole_pc)
    # aux = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.array(part_pc.points).mean(axis=0))
    # coord.vertices = Vector3dVector(np.array(aux.vertices))

    if not render_setup:
        points = [[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5],
                  [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5]]
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        vis.add_geometry(line_set)
        # vis.add_geometry(coord)
        vis.add_geometry(full_pcd)
        render_setup = True

    # vis.update_geometry(coord)
    vis.update_geometry(full_pcd)

    vis.poll_events()
    vis.update_renderer()
    #
    # i = i+1
print(tot)
    # print('Rendering', time.time() - start)