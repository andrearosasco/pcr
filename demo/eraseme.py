import copy
import time
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from torch import cdist
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD

from configs import ModelConfig, TrainConfig
from main import HyperNetwork
from pathlib import Path

import numpy as np
from PIL import Image
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
import torch
import open3d as o3d
from utils.input import RealSense
from utils.misc import create_3d_grid, check_mesh_contains

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
model = HyperNetwork.load_from_checkpoint('C:/Users/sberti/PycharmProjects/pcr/checkpoint/best', config=ModelConfig)
model = model.to('cuda')
model.eval()

# camera = RealSense()

#####################################################
############# Point Cloud Processing ################
#####################################################

### Read the pointcloud from a source
# depth = np.array(Image.open(f'000299-depth.png'), dtype=np.float32)
# _, depth = camera.read()
read_time, seg_time, inf_time, ref_time = 0, 0, 0, 0
for i in range(100):  # tqdm(range(100)):

    # start = time.time()
    depth = np.array(Image.open(f'depth_test.png'), dtype=np.uint16)
    # read_time += (time.time() - start)

    start = time.time()
    ### Cut the depth at a 2m distance
    depth[depth > 2000] = 0
    full_pc = RealSense.pointcloud(depth)

    ### Randomly subsample 5% of the total points (to ease DBSCAN processing)
    if i == 0:
        old_idx = np.random.choice(full_pc.shape[0], (int(full_pc.shape[0] * 0.05)), replace=False)
    idx = old_idx
    downsampled_pc = full_pc[idx]

    ### Apply DBSCAN and keep only the closest cluster to the camera
    clustering = DBSCAN(eps=0.1, min_samples=10).fit(downsampled_pc)
    close = clustering.labels_[downsampled_pc.argmax(axis=0)[2]]
    segmented_pc = downsampled_pc[clustering.labels_ == close]

    ### Randomly choose 2024 points (model input size)
    idx = np.random.choice(segmented_pc.shape[0], (2024), replace=False)
    size_pc = segmented_pc[idx]
    # seg_time += time.time() - start

    # start = time.time()
    ### Normalize Point Cloud
    mean = np.mean(size_pc, axis=0)
    var = np.sqrt(np.max(np.sum((size_pc - mean) ** 2, axis=1)))
    normalized_pc = (size_pc - mean) / (var * 2)

    # TODO START REMOVE DEBUG
    # partial_pcd = PointCloud()
    # partial_pcd.points = Vector3dVector(partial)
    # partial_pcd.paint_uniform_color([0, 1, 0])
    # coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries([partial_pcd, line_set])
    # TODO END REMOVE DEBUG

    ##################################################
    ################## Inference #####################
    ##################################################
    model_input = torch.FloatTensor(normalized_pc).unsqueeze(0).to(TrainConfig.device)
    samples = create_3d_grid(batch_size=model_input.shape[0], step=0.01).to(TrainConfig.device)

    fast_weights, _ = model.backbone(model_input)
    prediction = torch.sigmoid(model.sdf(samples, fast_weights))

    prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()

    # inf_time += time.time() - start

    ##################################################
    ################## Refinement ####################
    ##################################################
    start = time.time()
    refined_pred = torch.tensor(samples[:, prediction >= 0.5, :].cpu().detach().numpy(), device=TrainConfig.device,
                                requires_grad=True)
    # print(refined_pred.shape)
    refined_pred_0 = copy.deepcopy(refined_pred.detach())

    loss_function = BCEWithLogitsLoss(reduction='mean')
    optim = SGD([refined_pred], lr=0.5, momentum=0.9)

    c1, c2, c3 = 1, 0, 0 #1, 0, 0  1, 1e3, 0 # 0, 1e4, 5e2
    for step in range(10):
        results = model.sdf(refined_pred, fast_weights)

        gt = torch.ones_like(results[..., 0], dtype=torch.float32)
        gt[:, :] = 1
        loss1 = c1 * loss_function(results[..., 0], gt)
        loss2 = c2 * torch.mean((refined_pred - refined_pred_0) ** 2)
        loss3 = c3 * torch.mean(cdist(refined_pred, model_input).sort(dim=2)[0][:, :, 0])  # it works but it would be nicer to do the opposite
        loss_value = loss1 + loss2 + loss3

        model.zero_grad()
        optim.zero_grad()
        loss_value.backward(inputs=[refined_pred])
        optim.step()

        # print('Loss ', loss_value.item())

        # grad = refined_pred.grad.data
        # refined_pred = refined_pred - (1 * grad)

        # refined_pred = torch.tensor(refined_pred.cpu().detach().numpy(), device=TrainConfig.device,
        #                             requires_grad=True)

    ref_time += time.time() - start
    print(time.time() - start)
    ##################################################
    ################# Visualization ##################
    ##################################################
    selected = refined_pred.detach().cpu().squeeze().numpy()

    pred_pc = PointCloud()
    pred_pc.points = Vector3dVector(selected)
    pred_pc.paint_uniform_color([0, 0, 1])
    pred_pc.points = Vector3dVector(np.array(pred_pc.points) * (var * 2))
    t = np.eye(4)
    t[0:3, 3] = mean
    pred_pc.transform(t)

    part_pc = PointCloud()
    part_pc.points = Vector3dVector(full_pc)
    part_pc.paint_uniform_color([0, 1, 0])


# print(f'read time - {read_time / 100}')
# print(f'seg time - {seg_time / 100}')
# print(f'inf time - {inf_time / 100}')
print(f'ref time - {ref_time / 100}')
# print(f'tot time - {(read_time + seg_time + inf_time + ref_time) / 100}')

centers = o3d.geometry.LineSet()
centers.points = o3d.utility.Vector3dVector([np.mean(normalized_pc, axis=0).tolist(), mean.tolist(),
                                             np.mean(selected, axis=0).tolist(),
                                             np.mean(np.array(pred_pc.points), axis=0).tolist()])
centers.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3]])

o3d.visualization.draw_geometries([pred_pc, part_pc, line_set])