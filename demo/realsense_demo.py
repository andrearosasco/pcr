import copy
import time

import numpy
import onnx
import tqdm
from open3d import visualization
# from polygraphy.backend.trt import EngineFromBytes, TrtRunner
from sklearn.cluster import DBSCAN
from torch import cdist
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam

from configs import ModelConfig, TrainConfig
from production.inference import Infer, Refiner
from model import PCRNetwork
from pathlib import Path

import numpy as np
from PIL import Image

from production.tiny_tensor_rt import BackBone
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
from utils.misc import create_3d_grid, check_mesh_contains
import onnxruntime as ort

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
depth = np.array(Image.open(f'assets/depth_test.png'), dtype=np.uint16)
read_time += (time.time() - start)

start = time.time()
### Cut the depth at a 2m distance
depth[depth > 2000] = 0
full_pc = RealSense.pointcloud(depth)

### Randomly subsample 5% of the total points (to ease DBSCAN processing)
# if i == 0:
#     old_idx = np.random.choice(full_pc.shape[0], (int(full_pc.shape[0] * 0.05)), replace=False)
# idx = old_idx
idx = np.random.choice(full_pc.shape[0], (int(full_pc.shape[0] * 0.05)), replace=False)
downsampled_pc = full_pc[idx]

### Apply DBSCAN and keep only the closest cluster to the camera
clustering = DBSCAN(eps=0.1, min_samples=10).fit(downsampled_pc)
close = clustering.labels_[downsampled_pc.argmax(axis=0)[2]]
segmented_pc = downsampled_pc[clustering.labels_ == close]

### Randomly choose 2024 points (model input size)
if segmented_pc.shape[0] > 2024:
    idx = np.random.choice(segmented_pc.shape[0], (2024), replace=False)
    size_pc = segmented_pc[idx]
else:
    print('Warning: Partial Point Cloud padded')
    diff = 2024 - segmented_pc.shape[0]
    pad = np.zeros([diff, 3])
    pad[:] = segmented_pc[0]
    size_pc = np.vstack((segmented_pc, pad))
    # size_pc = segmented_pc
seg_time += time.time() - start

### Normalize Point Cloud
mean = np.mean(size_pc, axis=0)
var = np.sqrt(np.max(np.sum((size_pc - mean) ** 2, axis=1)))
normalized_pc = (size_pc - mean) / (var * 2)
normalized_pc[..., -1] = -normalized_pc[..., -1]

it = 1000
for i in tqdm.tqdm(range(it)):


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
    start = time.time()
    model_input = torch.FloatTensor(normalized_pc).unsqueeze(0).to(TrainConfig.device)
    # samples = create_3d_grid(batch_size=model_input.shape[0], step=0.01).to(TrainConfig.device)
    # samples = torch.randn(1, 1024, 3).to(TrainConfig.device)
    # with autocast():

    # with torch.no_grad():
    #     fast_weights, _ = model.backbone(model_input) # TODO questa riga funzione ma nel codice di backbone, fast_weights è il secondo argomento

    fast_weights = backbone(normalized_pc)
    # prediction = torch.sigmoid(model.sdf(samples, fast_weights))

    # outputs = backbone.run(None, {'input': np.expand_dims(normalized_pc, 0).astype(np.float32)})
    # *weights, _ = outputs
    # weights = [torch.tensor(w).cuda() for w in weights]
    # fast_weights = [[weights[i], weights[i + 1], weights[i + 2]] for i in range(0, 12, 3)]

    # with TrtRunner(engine) as runner:
    #     outputs = runner.infer(feed_dict={'input': normalized_pc})
    # res = list(outputs.values())
    # _, *weights = res
    # weights = [torch.tensor(w).cuda() for w in weights]
    # fast_weights = [[weights[i], weights[i + 1], weights[i + 2]] for i in range(0, 12, 3)]

        # prediction = torch.sigmoid(model.sdf(samples, fast_weights))
    #
    # prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()

    inf_time += time.time() - start

    ##################################################
    ################## Refinement ####################
    ##################################################
    start = time.time()
    # refined_pred = torch.tensor(samples[:, prediction.squeeze() >= 0.5, :].cpu().detach().numpy(), device=TrainConfig.device,
    #                             requires_grad=True)
    refined_pred = torch.tensor(torch.randn(1, 10000, 3).cpu().detach().numpy() * 1, device=TrainConfig.device,
                                requires_grad=True)
    # refined_pred_0 = copy.deepcopy(refined_pred.detach())

    loss_function = BCEWithLogitsLoss(reduction='mean')
    optim = Adam([refined_pred], lr=0.1)

    # complete = []
    # selected = refined_pred.detach().cpu().numpy()
    # for _ in range(10):
    #     selected, idx = refiner(selected, fast_weights)
    #     complete.append(selected.reshape(1, 10000, 3)[:, idx, :])

    c1, c2, c3, c4 = 1, 0, 0, 0 #1, 0, 0  1, 1e3, 0 # 0, 1e4, 5e2
    new_points = [] # refined_pred.detach().clone()
    for step in range(20):
        results = model.sdf(refined_pred, fast_weights)
        new_points += [refined_pred.detach().clone()[:, (torch.sigmoid(results).squeeze() >= 0.5) * (torch.sigmoid(results).squeeze() <= 1), :]]

        gt = torch.ones_like(results[..., 0], dtype=torch.float32)
        gt[:, :] = 1
        loss1 = c1 * loss_function(results[..., 0], gt)
        # loss2 = c2 * torch.mean((refined_pred - refined_pred_0) ** 2)
        # loss3 = c3 * torch.mean(
        #     cdist(refined_pred, model_input).sort(dim=1)[0][:, :100, :])  # it works but it would be nicer to do the opposite
        # loss4 = c4 * (1/(cdist(refined_pred, refined_pred)+1e-5)).max(dim=1)[0].sum()
        loss_value = loss1 #+ loss2 + loss3 + loss4

        model.zero_grad()
        optim.zero_grad()
        loss_value.backward(inputs=[refined_pred])
        optim.step()

        # print('Loss ', loss_value.item())

        # grad = refined_pred.grad.data
        # refined_pred = refined_pred - (1 * refined_pred.grad.data)

        # refined_pred = torch.tensor(refined_pred.cpu().detach().numpy(), device=TrainConfig.device,
        #                             requires_grad=True)
    # selected = numpy.concatenate(complete, axis=1)

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


print(f'read time - {read_time / it} s -> {1 / (read_time / it)} fps')
print(f'seg time - {seg_time / it} s -> {1 / (seg_time / it)} fps')
print(f'inf time - {inf_time / it} s -> {1 / (inf_time / it)} fps')
print(f'ref time - {ref_time / it} s -> {1 / (ref_time / it)} fps')
print(f'tot time - {(read_time + seg_time + inf_time + ref_time) / it}')


viewer.create_window(width=1920, height=1080, visible=True)
viewer.clear_geometries()

viewer.add_geometry(pred_pc)
viewer.add_geometry(part_pc)
viewer.add_geometry(line_set)

control = viewer.get_view_control()
control.set_front(np.array([0.2, 0.1, 0.05]))
control.set_lookat(np.array([0, 0, 0]))
control.set_up(np.array([0, 1, 0]))
control.set_zoom(1)

viewer.run()

# depth = viewer.capture_screen_image(f'new_method10.png', True)

viewer.remove_geometry(pred_pc)
viewer.remove_geometry(part_pc)
viewer.destroy_window()

centers = o3d.geometry.LineSet()
centers.points = o3d.utility.Vector3dVector([np.mean(normalized_pc, axis=0).tolist(), mean.tolist(),
                                             np.mean(selected, axis=0).tolist(),
                                             np.mean(np.array(pred_pc.points), axis=0).tolist()])
centers.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3]])
#
# o3d.visualization.draw_geometries([pred_pc, part_pc, line_set])
