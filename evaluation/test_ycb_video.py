import os
import copy
import open3d as o3d
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from sklearn.cluster import DBSCAN
from configs import server_config
from model.PCRNetwork import PCRNetwork as Model
from utils.input import YCBVideoReader
import numpy as np
from utils.misc import from_depth_to_pc
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    box_cls = {"003_cracker_box": 2,
               "004_sugar_box": 3,
               "008_pudding_box": 7,
               "009_gelatin_box": 8,
               "036_wood_block": 16,
               "061_foam_brick": 21}
    reader = YCBVideoReader("assets/fake_YCB")
    while True:
        _ = reader.get_frame()
        if _ is None:  # Dataset is over
            break
        frame_path, boxes, rgb, depth, label, meta, intrinsics = _
        for i, obj_name in enumerate(boxes.keys()):
            if obj_name not in box_cls.keys():
                continue

            # Reconstruct point cloud
            obj_depth = copy.deepcopy(depth)
            obj_depth[label != box_cls[obj_name]] = 0.
            points = from_depth_to_pc(obj_depth, list(intrinsics.values()), float(meta["factor_depth"]))

            # Remove outlier
            # pc = PointCloud()  # TODO REMOVE DEBUG
            # pc.points = Vector3dVector(points)  # TODO REMOVE DEBUG
            # o3d.visualization.draw_geometries([pc])  # TODO REMOVE DEBUG
            good = DBSCAN(eps=0.01, min_samples=100).fit(points).core_sample_indices_
            points = points[good]
            # pc = pc.select_by_index(good)  # TODO REMOVE DEBUG
            # o3d.visualization.draw_geometries([pc])  # TODO REMOVE DEBUG

            # Normalize point cloud
            mean = np.mean(points, axis=0)
            points = points - mean
            var = np.sqrt(np.max(np.sum(points ** 2, axis=1)))
            points = points / (var * 2)  #(var * 2) (1040*2)

            # Load model
            model = Model.load_from_checkpoint('./checkpoint/final', config=server_config.ModelConfig)
            model.cuda()
            model.eval()

            # TODO normalize point cloud and give it to the model
            indices = torch.randperm(len(points))[:2048]
            points = points[indices]
            points_tensor = torch.FloatTensor(points).unsqueeze(0).cuda()
            res = model(points_tensor)
            res = res.detach().cpu().numpy()

            # TODO REMOVE DEBUG SHOW
            partial = PointCloud()
            partial.points = Vector3dVector(points)
            # partial.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            complete = PointCloud()
            complete.points = Vector3dVector(res)
            # complete.transform(t)
            # complete.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            complete.paint_uniform_color([255, 0, 0])
            partial.paint_uniform_color([0, 255, 0])

            o3d.visualization.draw_geometries([partial, complete, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)])

            # Move full point cloud to partial point cloud
            obj_xyz = reader.get_xyz_by_name(obj_name)
            obj_xyz = (obj_xyz @ meta["poses"][..., i][:, :3].T)
            obj_xyz += meta["poses"][..., i][:, -1]
            # t = np.vstack([meta['poses'][:, :, i], np.eye(4)[3, :]])

            # Load model
            model = Model.load_from_checkpoint('./checkpoint/latest', config=server_config.ModelConfig)
            model.cuda()
            model.eval()

            # TODO normalize point cloud and give it to the model
            indices = torch.randperm(len(points))[:2048]
            points = points[indices]
            points_tensor = torch.FloatTensor(points).unsqueeze(0).cuda()
            res = model(points_tensor)

            # TODO REMOVE DEBUG
            partial = PointCloud()
            partial.points = Vector3dVector(points)
            # partial.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            complete = PointCloud()
            complete.points = Vector3dVector(obj_xyz)

            complete.paint_uniform_color([255, 0, 0])
            partial.paint_uniform_color([0, 255, 0])

            o3d.visualization.draw_geometries([partial, complete, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)])

            np.save("test1", points)  # TODO REMOVE DEBUG
            np.save("test2", obj_xyz)  # TODO REMOVE DEBUG
            print(os.getcwd())  # TODO REMOVE DEBUG
