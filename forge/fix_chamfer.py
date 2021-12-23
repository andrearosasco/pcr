import time

from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader

import sys
sys.argv[0] = 'server_config'
from utils.metrics import chamfer_distance
from configs import DataConfig, ModelConfig, TrainConfig
from datasets.BoxNetPOVDepth import BoxNet as Dataset
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
import torch
import open3d as o3d
import numpy as np

from models import HyperNetwork
from utils.misc import create_3d_grid, collate
import os

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

hands = [o3d.geometry.TriangleMesh.create_coordinate_frame() for _ in range(2)]

#####################################################
############# Model and Camera Setup ################
#####################################################
model = HyperNetwork.load_from_checkpoint('checkpoint/depth_best', config=ModelConfig)
model = model.to('cuda')
model.eval()

if __name__ == '__main__':
    #####################################################
    ############# Point Cloud Processing ################
    #####################################################
    samples = create_3d_grid(batch_size=1, step=0.01).to(TrainConfig.device)
    valid_set = Dataset(DataConfig, 10)

    dl = DataLoader(
                valid_set,
                shuffle=False,
                batch_size=1,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate)


    elapsed_time_prep, elapsed_time_inf, elapsed_time_vis, elapsed_time_data = 0, 0, 0, 0

    # start_time_data = time.time()
    # for label, partial, meshes, _, _ in dl:
    #     elapsed_time_data += time.time() - start_time_data
    #     a = partial.to('cuda')
    #     b = len(meshes)
    #     start_time_data = time.time()
    # print(elapsed_time_data / 100)

    start_time_data = time.time()
    for label, partial, meshes, _, _ in dl:
        elapsed_time_data = time.time() - start_time_data

        start_time_prep = time.time()
        model_input = partial.to(TrainConfig.device)
        verts, tris = meshes
        meshes_list = []
        verts, verts_lengths = pad_packed_sequence(verts, batch_first=True, padding_value=0.)
        tris, tris_lenths = pad_packed_sequence(tris, batch_first=True, padding_value=0.)
        for vert, l1, tri, l2 in zip(verts, verts_lengths, tris, tris_lenths):
            meshes_list.append(o3d.geometry.TriangleMesh(Vector3dVector(vert[:l1].cpu()),
                                                         Vector3iVector(tri[:l2].cpu())))
        elapsed_time_prep = (time.time() - start_time_prep)
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

        start_time_inf = time.time()
        fast_weights, _ = model.backbone(model_input)
        prediction = torch.sigmoid(model.sdf(samples, fast_weights))

        prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()
        selected = torch.tensor(samples[:, prediction >= 0.5, :].cpu().detach().numpy(), device=TrainConfig.device)

        elapsed_time_inf = (time.time() - start_time_inf)
        ##################################################
        ################# Visualization ##################
        ##################################################

        start_time_vis = time.time()
        print(chamfer_distance(selected, meshes_list))

        # pred_pc = PointCloud()
        # pred_pc.points = Vector3dVector(selected.detach().cpu().squeeze().numpy())
        # pred_pc.paint_uniform_color([0, 0, 1])
        #
        # full_pc = meshes_list[0].sample_points_uniformly(10000)
        # full_pc.paint_uniform_color([0, 1, 0])
        #
        # part_pc = PointCloud()
        # part_pc.points = Vector3dVector(partial.squeeze().cpu().numpy())
        # part_pc.paint_uniform_color([1, 0, 0])
        elapsed_time_vis = (time.time() - start_time_vis)

        print(elapsed_time_data)
        print(elapsed_time_prep)
        print(elapsed_time_inf)
        print(elapsed_time_vis)

        # o3d.visualization.draw_geometries([pred_pc, part_pc, line_set, full_pc])

        start_time_data = time.time()
