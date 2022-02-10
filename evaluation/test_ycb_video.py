import copy
import os
# try:
#     from open3d.cuda.pybind.geometry import PointCloud, TriangleMesh
# except ImportError:
from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh
from open3d.cpu.pybind.utility import Vector3dVector

os.environ['PYTHONPATH'] += ':' + '.'
from tqdm import tqdm
import open3d as o3d
import torch
from sklearn.cluster import DBSCAN
from main import HyperNetwork
from configs import ModelConfig
from utils.input import YCBVideoReader
import numpy as np
from utils.misc import from_depth_to_pc

# srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=4 --mem=40G bash

box_cls = {"003_cracker_box": 2,
           "004_sugar_box": 3,
           "008_pudding_box": 7,
           "009_gelatin_box": 8,
           "036_wood_block": 16,
           "061_foam_brick": 21}

distances = {"003_cracker_box": [],
             "004_sugar_box": [],
             "008_pudding_box": [],
             "009_gelatin_box": [],
             "036_wood_block": [],
             "061_foam_brick": []}


def array_mesh_chamfer(array, mesh):
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh)
    query_points = o3d.core.Tensor(array, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_distance(query_points)
    return np.mean(signed_distance.numpy())


if __name__ == "__main__":

    # Load data reader
    reader = YCBVideoReader('fake_YCB')

    # Count total number of file
    total = sum([len(files) for r, d, files in os.walk(reader.data_path)]) / 5
    total = total / reader.jump_n_frames

    # Load model
    # model = HyperNetwork.load_from_checkpoint('/home/IIT.LOCAL/arosasco/projects/pcr/checkpoint/absolute_best',
    #                                           config=ModelConfig)
    model = HyperNetwork.load_from_checkpoint('checkpoint/latest',
                                              config=ModelConfig)
    model.cuda()
    model.eval()

    # Iterate all frames
    with tqdm(total=total) as pbar:
        while True:
            _ = reader.get_frame()

            pbar.update()

            if _ is None:  # Dataset is over
                break

            frame_path, boxes, rgb, depth, label, meta, intrinsics = _

            # For each object
            for i, obj_name in enumerate(boxes.keys()):

                # If the object is not a box, skip
                if obj_name not in box_cls.keys():
                    continue

                # Reconstruct point cloud
                obj_depth = copy.deepcopy(depth)
                obj_depth[label != box_cls[obj_name]] = 0.
                points = from_depth_to_pc(obj_depth, list(intrinsics.values()), float(meta["factor_depth"]))

                # Remove outlier
                # pc = PointCloud()  # TODO REMOVE DEBUG
                # pc.points = Vector3dVector(points)  # TODO REMOVE DEBUG
                # o3d.visualization.draw_geometries([pc,
                #                                    o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)])
                good = DBSCAN(eps=0.01, min_samples=100).fit(points).core_sample_indices_
                points = points[good]
                # pc = pc.select_by_index(good)  # TODO REMOVE DEBUG
                # o3d.visualization.draw_geometries([pc])  # TODO REMOVE DEBUG

                # Normalize point cloud
                mean = np.mean(points, axis=0)
                points = points - mean
                var = np.sqrt(np.max(np.sum(points ** 2, axis=1)))
                points = points / (var * 2)

                # Random downsampling
                indices = torch.randperm(len(points))[:2048]
                points = points[indices]
                points_tensor = torch.FloatTensor(points).unsqueeze(0).cuda()

                # Give partial point cloud to the model and get full point cloud
                res = model(points_tensor)
                res = res.detach().cpu().numpy()

                # Move reconstructed point cloud in the original position
                # res = res * (var * 2)
                # res = res + mean

                ##########
                # PART 2 #
                ##########
                # Load model mesh
                obj_mesh_path = reader.get_mesh_path_by_name(obj_name)
                obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)

                # Rotate mesh vertices accordingly with the partial point cloud
                aux = np.array(obj_mesh.vertices) @ meta["poses"][..., i][:, :3].T  # rotate as partial
                aux += meta["poses"][..., i][:, -1]  # translate as partial
                aux = aux - mean  # translate as model input
                aux = aux / (var * 2)  # scale as model input
                obj_mesh.vertices = Vector3dVector(aux)

                # Compute Chamfer Distance
                d = array_mesh_chamfer(res, obj_mesh)
                print(d)

                # TODO REMOVE DEBUG
                aux = np.array(obj_mesh.vertices)
                aux = aux * (var * 2)
                aux = aux + mean
                obj_mesh.vertices = Vector3dVector(aux)

                res = res * (var * 2)
                res = res + mean
                complete = PointCloud()
                complete.points = Vector3dVector(res)
                complete.paint_uniform_color([1, 0, 0])

                partial = obj_mesh.sample_points_uniformly(len(complete.points))
                partial.paint_uniform_color([0, 1, 0])

                d = array_mesh_chamfer(res, obj_mesh)
                print(d)

                o3d.visualization.draw_geometries([complete, partial,
                                                   o3d.geometry.TriangleMesh.create_coordinate_frame()])
                # TODO REMOVE DEBUG

                distances[obj_name].append(d)

    for elem in distances.keys():
        if len(distances[elem]) > 0:
            distances[elem] = sum(distances[elem]) / len(distances[elem])
        else:
            distances[elem] = "Zero elements found"

    print(distances)
