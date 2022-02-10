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
import open3d as o3d

from configs import ModelConfig, TrainConfig
from model import PCRNetwork
from utils.input import RealSense
from utils.misc import create_3d_grid

model = PCRNetwork.load_from_checkpoint('./checkpoint/absolute_best', config=ModelConfig)
model = model.to('cuda')
model.eval()


vis = Visualizer()
vis.create_window('Pose Estimation')
camera = RealSense(640, 480)

full_pcd = PointCloud()
render_setup = False
i = 0
device = 'cuda'


while True:

    rgb, depth = camera.read()
    # rgb, depth = np.array(PIL.Image.open('color.png'))[..., ::-1], np.array(PIL.Image.open('depth.png'), dtype=np.float32)

    cv2.imshow('rgb', rgb)
    cv2.imshow('depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))
    cv2.waitKey(1)


    # rgb, depth = camera.read()
    #
    # cv2.imshow('depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))
    # cv2.waitKey(0)
    #
    # clustering.labels_
    # histg = cv2.calcHist([depth], [0], None, [np.max(depth)], [0, np.max(depth)])
    # plt.plot(histg[1:])
    # plt.show()
    #
    # print('Read', time.time() - start)
    # start = time.time()

    depth[depth > 2000] = 0
    pc = camera.pointcloud(depth)
    pcd = PointCloud()
    pcd.points = Vector3dVector(pc)

    whole_pc = copy.deepcopy(pcd)

    idx = np.random.choice(pc.shape[0], (int(pc.shape[0] * 0.05)), replace=False)
    pc = pc[idx]
    clustering = DBSCAN(eps=0.1, min_samples=10).fit(pc)
    close = clustering.labels_[pc.argmax(axis=0)[2]]
    pcd.points = Vector3dVector(pc[clustering.labels_ == close])
    pcd.paint_uniform_color([0, 1, 0])

    pc = pc[clustering.labels_ == close]

    seg_pcd = PointCloud()
    seg_pcd.points = Vector3dVector(pc)
    seg_pcd.paint_uniform_color([0, 0, 1])

    if pc.shape[0] > 2048:
        idx = np.random.choice(pc.shape[0], (2048), replace=False)
        pc = pc[idx]
    else:
        diff = 2048 - pc.shape[0]
        partial_pcd = np.vstack((pc, np.zeros([diff, 3])))  # TODO nooooooo

    partial = torch.FloatTensor(pc)  # Must be 1, 2024, 3

    # Normalize Point Cloud as training time
    partial = np.array(partial)
    mean = np.mean(np.array(partial), axis=0)
    partial = np.array(partial) - mean
    var = np.sqrt(np.max(np.sum(partial ** 2, axis=1)))
    partial = partial / (var * 2)

    # partial[..., -1] = -partial[..., -1]  # TODO VERIFY

    # TODO START REMOVE DEBUG
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries([partial_pcd])
    # TODO END REMOVE DEBUG

    # Inference
    partial = torch.FloatTensor(partial).unsqueeze(0).to(device)

    samples = create_3d_grid(batch_size=partial.shape[0], step=0.01).to(TrainConfig.device)  # TODO we create grid two times...

    fast_weights, _ = model.backbone(partial)  # TODO step SHOULD ME 0.01
    prediction = torch.sigmoid(model.sdf(samples, fast_weights))

    prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()
    samples = samples.squeeze(0).detach().cpu().numpy()

    selected = samples[prediction > 0.5]

    partial = partial.squeeze(0).detach().cpu().numpy()
    part_pc = PointCloud()
    part_pc.points = Vector3dVector(partial)
    colors = np.array([0, 255, 0])[None, ...].repeat(partial.shape[0], axis=0)
    part_pc.colors = Vector3dVector(colors)

    pred_pc = PointCloud()
    pred_pc.points = Vector3dVector(selected)
    colors = np.array([0, 0, 255])[None, ...].repeat(selected.shape[0], axis=0)
    pred_pc.colors = Vector3dVector(colors)



    # pc.colors = Vector3dVector(palette[clustering.labels_])
    part_pc.paint_uniform_color([1, 0, 0])
    full_pcd.clear()
    pred_pc.points = Vector3dVector((np.array(pred_pc.points) + mean) * (var * 2))
    pred_pc.paint_uniform_color([1, 0, 0])
    whole_pc.paint_uniform_color([0, 1, 0])
    full_pcd += (pred_pc + whole_pc + seg_pcd)

    # print('Conversion', time.time() - start)
    # start = time.time()

    if not render_setup:
        points = [[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5],
                  [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5]]
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # vis.add_geometry(line_set)
        vis.add_geometry(full_pcd)
        render_setup = True

    vis.update_geometry(full_pcd)

    vis.poll_events()
    vis.update_renderer()
    #
    # i = i+1

    # print('Rendering', time.time() - start)