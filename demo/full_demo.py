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
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD

from configs.server_config import ModelConfig, TrainConfig
from frompartialtopose import FromPartialToPose
from main import HyperNetwork
from utils.input import RealSense
from utils.misc import create_3d_grid

#####################################################
########## Output/Input Space Boundaries ############
#####################################################
points = [[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5],
          [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5]]
lines = [[0, 1], [0, 2], [1, 3], [2, 3],
         [4, 5], [4, 6], [5, 7], [6, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)

#####################################################
############# Model and Camera Setup ################
#####################################################
model = HyperNetwork.load_from_checkpoint('../checkpoint/depth_best', config=ModelConfig)
model = model.to('cuda')
model.eval()

vis = Visualizer()
vis.create_window('Pose Estimation')
camera = RealSense(640, 480)

render_pcd = PointCloud()
hands = [o3d.geometry.TriangleMesh.create_coordinate_frame() for _ in range(2)]

render_setup = False
i = 0
device = 'cuda'

#####################################################
############# Point Cloud Processing ################
#####################################################

while True:
    ### Read the pointcloud from a source
    rgb, depth = camera.read()

    cv2.imshow('rgb', rgb)
    cv2.imshow('depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))
    cv2.waitKey(1)

    ### Cut the depth at a 2m distance
    depth[depth > 2000] = 0
    full_pc = RealSense.pointcloud(depth)

    ### Randomly subsample 5% of the total points (to ease DBSCAN processing)
    idx = np.random.choice(full_pc.shape[0], (int(full_pc.shape[0] * 0.05)), replace=False)
    downsampled_pc = full_pc[idx]

    ### Apply DBSCAN and keep only the closest cluster to the camera
    clustering = DBSCAN(eps=0.1, min_samples=10).fit(downsampled_pc)
    close = clustering.labels_[downsampled_pc.argmax(axis=0)[2]]
    segmented_pc = downsampled_pc[clustering.labels_ == close]

    ### Randomly choose 2024 points (model input size)
    if segmented_pc.shape[0] > 2048:
        idx = np.random.choice(segmented_pc.shape[0], (2048), replace=False)
        size_pc = segmented_pc[idx]
    else:
        size_pc = segmented_pc

    ### Normalize Point Cloud
    mean = np.mean(size_pc, axis=0)
    var = np.sqrt(np.max(np.sum((size_pc - mean) ** 2, axis=1)))
    normalized_pc = (size_pc - mean) / (var * 2)
    normalized_pc[..., -1] = -normalized_pc[..., -1]

    ##################################################
    ################## Inference #####################
    ##################################################
    model_input = torch.FloatTensor(normalized_pc).unsqueeze(0).to(TrainConfig.device)
    samples = create_3d_grid(batch_size=model_input.shape[0], step=0.01).to(TrainConfig.device)

    fast_weights, _ = model.backbone(model_input)
    prediction = torch.sigmoid(model.sdf(samples, fast_weights))

    prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()

    ##################################################
    ################## Refinement ####################
    ##################################################
    start = time.time()
    refined_pred = torch.tensor(samples[:, prediction >= 0.5, :].cpu().detach().numpy(), device=TrainConfig.device,
                                requires_grad=True)
    refined_pred_0 = copy.deepcopy(refined_pred.detach())

    loss_function = BCEWithLogitsLoss(reduction='mean')
    optim = SGD([refined_pred], lr=0.01, momentum=0.9)

    c1, c2, c3 = 1, 0, 0  # 1, 0, 0  1, 1e3, 0 # 0, 1e4, 5e2
    for step in range(0):
        results = model.sdf(refined_pred, fast_weights)

        gt = torch.ones_like(results[..., 0], dtype=torch.float32)
        gt[:, :] = 1
        loss1 = c1 * loss_function(results[..., 0], gt)
        # loss2 = c2 * torch.mean((refined_pred - refined_pred_0) ** 2)
        # loss3 = c3 * torch.mean(
        #     cdist(refined_pred, model_input).sort(dim=1)[0][:, :100, :])  # it works but it would be nicer to do the opposite
        loss_value = loss1  # + loss2 + loss3

        model.zero_grad()
        optim.zero_grad()
        loss_value.backward(inputs=[refined_pred])
        optim.step()

        print('Loss ', loss_value.item())

    ##################################################
    ################# Visualization ##################
    ##################################################

    selected = refined_pred.detach().cpu().squeeze().numpy()
    pred_pc = PointCloud()
    pred_pc.points = Vector3dVector(selected)
    pred_pc.paint_uniform_color([0, 0, 1])

    part_pc = PointCloud()
    part_pc.points = Vector3dVector(normalized_pc)
    part_pc.paint_uniform_color([0, 1, 0])

    # res = FromPartialToPose(None, grid_res=0.01).find_poses(pred_pc)
    # c1, n1, c2, n2 = res
    # r1, r2 = FromPartialToPose.get_rotations(res)
    #
    # for c, r, n, h in zip([c1, c2], [r1, r2], [n1, n2], hands):
    #     c[2] = -c[2]  # Invert depth
    #     n[2] = -n[2]  # Invert normal in depth dimension
    #     c = c * var * 2  # De-normalize center
    #
    #     aux_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    #     aux_mesh.translate(c, relative=False)
    #     R = FromPartialToPose.create_rotation_matrix(np.array([0, 0, 1]), n)
    #     aux_mesh.rotate(R, center=c)
    #     aux_mesh.translate(mean, relative=True)
    #
    #     aux_mesh.translate(c, relative=False)
    #     aux_mesh.rotate(r, center=c)
    #     aux_mesh.translate(mean, relative=True)  # Translate coord as the point cloud
    #
    #     h.triangles = aux_mesh.triangles
    #     h.vertices = aux_mesh.vertices


    render_pcd.clear()
    render_pcd += (pred_pc + part_pc)

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

        # for h in hands:
        #     vis.add_geometry(h)
        vis.add_geometry(line_set)
        vis.add_geometry(render_pcd)
        render_setup = True

    vis.update_geometry(render_pcd)
    # for h in hands:
    #     vis.update_geometry(h)

    vis.poll_events()
    vis.update_renderer()
    #
    # i = i+1

    # print('Rendering', time.time() - start)