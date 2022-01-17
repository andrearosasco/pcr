import os
import copy

import cv2
import open3d as o3d
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from sklearn.cluster import DBSCAN
from configs import server_config
from main import HyperNetwork
from utils.input import YCBVideoReader
import numpy as np
from utils.misc import from_depth_to_pc
from scipy.spatial.transform import Rotation as R


def project( rgb, points):
    k = np.eye(3)
    k[0, :] = np.array([1066.778, 0, 312.9869])
    k[1, 1:] = np.array([1067.487, 241.3109])

    points = np.array(points) * 10000.0
    uv = k @ points.T
    uv = uv[0:2] / uv[2, :]

    uv = np.round(uv, 0).astype(int)

    uv[0, :] = np.clip(uv[0, :], 0, 639)
    uv[1, :] = np.clip(uv[1, :], 0, 479)

    rgb[uv[1, :], uv[0, :], :] = np.tile((np.array([1, 0, 0]) * 255).astype(int), (uv.shape[1], 1))

    return rgb

if __name__ == "__main__":
    box_cls = {"003_cracker_box": 2,
               "004_sugar_box": 3,
               "008_pudding_box": 7,
               "009_gelatin_box": 8,
               "036_wood_block": 16,
               "061_foam_brick": 21}
    reader = YCBVideoReader("assets/fake_YCB")
    j = 0
    while True:
        _ = reader.get_frame()
        if _ is None:  # Dataset is over
            break
        frame_path, boxes, rgb, depth, label, meta, intrinsics = _
        for i, obj_name in enumerate(boxes.keys()):
            if obj_name not in box_cls.keys():
                continue

            j += 1
            # Reconstruct point cloud
            obj_depth = copy.deepcopy(depth)
            obj_depth[label != box_cls[obj_name]] = 0.
            partial_pc = from_depth_to_pc(obj_depth, list(intrinsics.values()), float(meta["factor_depth"]))

            # Remove outlier
            # pc = PointCloud()  # TODO REMOVE DEBUG
            # pc.points = Vector3dVector(points)  # TODO REMOVE DEBUG
            # o3d.visualization.draw_geometries([pc])  # TODO REMOVE DEBUG
            idx = DBSCAN(eps=0.01, min_samples=100).fit(partial_pc).core_sample_indices_
            clean_pc = partial_pc[idx]
            # pc = pc.select_by_index(good)  # TODO REMOVE DEBUG
            # o3d.visualization.draw_geometries([pc])  # TODO REMOVE DEBUG

            # Normalize point cloud
            mean = np.mean(clean_pc, axis=0)
            points = clean_pc - mean
            var = np.sqrt(np.max(np.sum(points ** 2, axis=1)))
            points = points / (var * 2)  #(var * 2) (1040*2)

            # Load model
            model = HyperNetwork.load_from_checkpoint('./checkpoint/final', config=server_config.ModelConfig)
            model.cuda()
            model.eval()

            # TODO normalize point cloud and give it to the model
            indices = torch.randperm(len(points))[:2048]
            downsampled_pc = points[indices]
            points_tensor = torch.FloatTensor(downsampled_pc).unsqueeze(0).cuda()
            res = model(points_tensor)
            res = res.detach().cpu().numpy()

            # TODO REMOVE DEBUG SHOW
            partial = PointCloud()
            partial.points = Vector3dVector(downsampled_pc)
            # partial.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            complete = PointCloud()
            complete.points = Vector3dVector(res)

            # complete.transform(t)
            # complete.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            complete.paint_uniform_color([255, 0, 0])
            partial.paint_uniform_color([0, 255, 0])

            aux = PointCloud()
            aux.points = Vector3dVector(np.array(complete.points) * (var * 2))
            t = np.eye(4)
            t[0:3, 3] = mean
            aux.transform(t)
            # idx = np.random.choice(np.array(aux.points).shape[0], 2500, replace=False)
            res = project(rgb, np.array(aux.points))
            cv2.imshow('projection', res)
            cv2.imwrite(f'proj_{j}.png', res)

            complete.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            partial.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            o3d.visualization.draw_geometries([partial, complete])
            cv2.waitKey(0)
            # t = np.vstack([meta['poses'][:, :, i], np.eye(4)[3, :]])
