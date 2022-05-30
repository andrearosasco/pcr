import copy
import time

import cv2
import numpy
import onnx
import tqdm
from open3d import visualization
# from polygraphy.backend.trt import EngineFromBytes, TrtRunner
from sklearn.cluster import DBSCAN
from torch import cdist, nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam
from torchvision import models

from configs import ModelConfig, TrainConfig
from production.inference import Infer, Refiner
from model import PCRNetwork
from pathlib import Path

import numpy as np
from PIL import Image

from production.tiny_tensor_rt import BackBone
from utils.Segmentator import Segmentator
from utils.reproducibility import make_reproducible

try:
    from open3d.cuda.pybind.geometry import PointCloud
    from open3d.cuda.pybind.utility import Vector3dVector
    from open3d.cuda.pybind.visualization import draw_geometries

except ImportError:
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector
    from open3d.cpu.pybind.visualization import draw_geometries
import torch
import open3d as o3d
from utils.input import RealSense
from utils.misc import create_3d_grid, check_mesh_contains, project_pc, pc_to_depth
import onnxruntime as ort

def draw_pc(*pcs):
    o3d_pcs = []
    for pc in pcs:
        aux = PointCloud()
        aux.points = Vector3dVector(pc)
        o3d_pcs.append(aux)

    draw_geometries(o3d_pcs)

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

make_reproducible(1)

#####################################################
############# Model and Camera Setup ################
#####################################################
model = PCRNetwork.load_from_checkpoint('checkpoint/final', config=ModelConfig)

model = model.to('cuda')
model.eval()

# backbone = ort.InferenceSession('pcr.onnx')

# backbone = BackBone()
backbone = Infer('pcr.engine')
refiner = Refiner('refiner.engine')
# with open('assets/production/pcr.engine', 'rb') as f:
#     serialized_engine = f.read()
#     engine = EngineFromBytes(serialized_engine)


seg = models.segmentation.fcn_resnet101(pretrained=False)
seg.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
seg.load_state_dict(torch.load('../Segmentation/checkpoints/sym5/epoch23'), strict=False)
seg.eval()

seg = Segmentator(seg, device='cuda')


# camera = RealSense()
viewer = visualization.Visualizer()
#####################################################
############# Point Cloud Processing ################
#####################################################

### Read the pointcloud from a source
# depth = np.array(Image.open(f'000299-depth.png'), dtype=np.float32)
# _, depth = camera.read()
read_time, seg_time, inf_time, ref_time = 0, 0, 0, 0

start = time.time()
for i in range(2, 22):

    frame = np.array(Image.open(f'../Segmentation/data/real/data/rgb/eCub{i}_rgb.png'))
    depth = np.array(Image.open(f'../Segmentation/data/real/data/depth/eCub{i}_depth.png')).astype(np.uint16)

    cv2.imshow('rgb', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imshow('depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))
    key = cv2.waitKey(0)

    if key == ord('s'):
        continue

    read_time += (time.time() - start)

    start = time.time()

    x = seg.preprocess(frame).to(device='cuda')
    with torch.no_grad():
        y = seg.model(x)['out']
    score, categories = seg.postprocess(y)

    ## DEBUG ##

    _, small_categories = seg.postprocess(y, width=256, height=192)
    small_frame = cv2.resize(frame, dsize=(256, 192), interpolation=cv2.INTER_LINEAR)
    small_frame[small_categories != 1] = np.array([0, 0, 0])
    resized = cv2.resize(small_frame, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)

    test_frame = copy.deepcopy(frame)
    test_frame[categories != 1] = np.array([0, 0, 0])

    cv2.imshow('test_seg1', cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    cv2.imshow('test_seg2', cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))
    
    cv2.waitKey()

    ## END ##

    segmented_depth = copy.deepcopy(depth)
    segmented_depth = cv2.resize(segmented_depth, dsize=(256, 192), interpolation=cv2.INTER_LINEAR)
    segmented_depth[seg.postprocess(y, width=256, height=192)[1] != 1] = 0.0

    segmented_depth = cv2.resize(segmented_depth, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    # segmented_depth[categories != 1] = 0.0

    segmented_pc = RealSense.pointcloud(segmented_depth, scale=1)

    overlay = copy.deepcopy(frame)
    if np.any(categories == 1):
        overlay[categories == 1] = np.array([0, 0, 128])
    res = cv2.addWeighted(frame, 1, overlay, 0.5, 0)

    cv2.imshow('segmented', cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    cv2.imshow('segmented_depth', cv2.applyColorMap(cv2.convertScaleAbs(segmented_depth, alpha=0.03), cv2.COLORMAP_JET))

    key = cv2.waitKey(0)
    if key == ord('s'):
        continue

    draw_pc(segmented_pc)

    ### Randomly subsample 5% of the total points (to ease DBSCAN processing)
    # if i == 0:
    #     old_idx = np.random.choice(full_pc.shape[0], (int(full_pc.shape[0] * 0.05)), replace=False)
    # idx = old_idx
    idx = np.random.choice(segmented_pc.shape[0], (int(segmented_pc.shape[0] * 0.05)), replace=False)
    downsampled_pc = segmented_pc[idx]

    draw_pc(downsampled_pc)

    aux = PointCloud()
    aux.points = Vector3dVector(downsampled_pc)
    cl, ind = aux.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)

    inlier_cloud = aux.select_by_index(ind)
    outlier_cloud = aux.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    denoised_pc = np.array(inlier_cloud.points)

    # denoised_depth = pc_to_depth(denoised_pc)
    #
    #
    # denoised_depth = cv2.resize(denoised_depth, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    #
    # denoised_pc = RealSense.pointcloud(denoised_depth, scale=1.0)

    ### Apply DBSCAN and keep only the closest cluster to the camera
    # clustering = DBSCAN(eps=0.1, min_samples=10).fit(downsampled_pc)
    # close = clustering.labels_[downsampled_pc.argmax(axis=0)[2]]
    # segmented_pc = downsampled_pc[clustering.labels_ == close]

    ### Randomly choose 2024 points (model input size)
    if denoised_pc.shape[0] == 0:
        continue
    elif denoised_pc.shape[0] > 2024:
        idx = np.random.choice(denoised_pc.shape[0], (2024), replace=False)
        size_pc = denoised_pc[idx]
    else:
        print('Info: Partial Point Cloud padded')
        diff = 2024 - denoised_pc.shape[0]
        pad = np.zeros([diff, 3])
        pad[:] = segmented_pc[0]
        size_pc = np.vstack((denoised_pc, pad))
    seg_time += time.time() - start

    draw_pc(size_pc)

    ### Normalize Point Cloud
    mean = np.mean(size_pc, axis=0)
    var = np.sqrt(np.max(np.sum((size_pc - mean) ** 2, axis=1)))
    normalized_pc = (size_pc - mean) / (var * 2)
    normalized_pc[..., -1] = -normalized_pc[..., -1]

    ##################################################
    ################## Inference #####################
    ##################################################
    start = time.time()
    model_input = torch.FloatTensor(normalized_pc).unsqueeze(0).to(TrainConfig.device)

    fast_weights = backbone(normalized_pc)

    inf_time += time.time() - start

    ##################################################
    ################## Refinement ####################
    ##################################################
    start = time.time()

    refined_pred = torch.tensor(torch.randn(1, 10000, 3).cpu().detach().numpy() * 1, device=TrainConfig.device,
                                requires_grad=True)

    loss_function = BCEWithLogitsLoss(reduction='mean')
    optim = Adam([refined_pred], lr=0.1)

    c1, c2, c3, c4 = 1, 0, 0, 0 #1, 0, 0  1, 1e3, 0 # 0, 1e4, 5e2
    new_points = [] # refined_pred.detach().clone()
    for step in range(20):
        results = model.sdf(refined_pred, fast_weights)
        new_points += [refined_pred.detach().clone()[:, (torch.sigmoid(results).squeeze() >= 0.5) * (torch.sigmoid(results).squeeze() <= 1), :]]

        gt = torch.ones_like(results[..., 0], dtype=torch.float32)
        gt[:, :] = 1
        loss1 = c1 * loss_function(results[..., 0], gt)

        loss_value = loss1

        model.zero_grad()
        optim.zero_grad()
        loss_value.backward(inputs=[refined_pred])
        optim.step()

    ref_time += time.time() - start
    ##################################################
    ################# Visualization ##################
    ##################################################
    selected = torch.cat(new_points, dim=1).cpu().squeeze().numpy()

    pred_pc = PointCloud()
    pred_pc.points = Vector3dVector(selected)
    pred_pc.paint_uniform_color([0, 0, 1])

    part_pc = PointCloud()
    part_pc.points = Vector3dVector(normalized_pc)
    part_pc.paint_uniform_color([0, 1, 0])


    draw_geometries([pred_pc, part_pc, line_set])

    centers = o3d.geometry.LineSet()
    centers.points = o3d.utility.Vector3dVector([np.mean(normalized_pc, axis=0).tolist(), mean.tolist(),
                                                 np.mean(selected, axis=0).tolist(),
                                                 np.mean(np.array(pred_pc.points), axis=0).tolist()])
    centers.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3]])

    normalized_pc[..., -1] = -normalized_pc[..., -1]

    aux = PointCloud()
    points = np.array(pred_pc.points)
    points[..., -1] = -points[..., -1]
    aux.points = Vector3dVector(points * (var * 2))
    t = np.eye(4)
    t[0:3, 3] = mean
    aux.transform(t)
    aux.paint_uniform_color([0, 1, 0.5])

    size_pcd = PointCloud()
    size_pcd.points = Vector3dVector(size_pc)
    size_pcd.paint_uniform_color([1, 0.5, 0])

    # draw_geometries([pred_pc, part_pc, aux, size_pcd])
    # idx = np.random.choice(np.array(aux.points).shape[0], 2500, replace=False)

    colored = RealSense.pointcloud(depth, frame)
    draw_geometries([colored, aux, size_pcd])

    # res = project_pc(frame[:, ::-1, :], np.array(aux.points))
    # cv2.imshow('projection_pred',  cv2.cvtColor(res[:, ::-1, :], cv2.COLOR_BGR2RGB))
    #
    # res = project_pc(frame[:, ::-1, :], size_pc)
    # cv2.imshow('projection_part', cv2.cvtColor(res[:, ::-1, :], cv2.COLOR_BGR2RGB))

    # cv2.waitKey(0)

    #
    # o3d.visualization.draw_geometries([pred_pc, part_pc, line_set])
