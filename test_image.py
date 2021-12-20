import msvcrt
import time
import open3d as o3d
import numpy as np
import torch
from PIL import Image

from frompartialtopose import FromPartialToPose
from main import HyperNetwork
from utils.icub_gazebo_interface import iCubGazebo
import cv2
from utils.input import RealSense
from utils.pose_visualizer import PoseVisualizer
from configs.server_config import ModelConfig
import pyrealsense2 as rs

try:
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cuda.pybind.visualization import draw_geometries, Visualizer
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    print("Open3d CUDA not found!")
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cpu.pybind.visualization import draw_geometries, Visualizer
    from open3d.cpu.pybind.geometry import PointCloud


device = "cuda"


if __name__ == "__main__":

    test = PoseVisualizer()
    icub = iCubGazebo()
    res = 0.01

    # Pose generator
    model = HyperNetwork.load_from_checkpoint('./checkpoint/from_depth', config=ModelConfig)
    model = model.to(device)
    model.eval()
    generator = FromPartialToPose(model, res, device)

    while True:
        # GET DEPTH IMAGE FROM GAZEBO AND CONVERT IT INTO A POINT CLOUD ################################################
        # Get image
        # rgb, depth = icub.read()
        # cv2.imshow('RGB', rgb)  # TODO VISUALIZE DEBUG
        # depth = cv2.imread("depth_test.png")

        # Get only red part
        # rgb_mask = rgb[..., 2] == 102  # Red is the last dimension
        # rgb_mask = rgb_mask.astype(float) * 255

        # Get only depth of the box
        # filtered_depth = np.where(rgb_mask, depth, 0.)
        # filtered_depth_img = filtered_depth.astype(float) * 255

        # Convert depth image to Point Cloud
        # fx = fy = 343.12110728152936
        # cx = 160.0
        # cy = 120.0
        # intrinsics = o3d.camera.PinholeCameraIntrinsic(320, 240, fx, fy, cx, cy)
        # partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(filtered_depth), intrinsics)
        # o3d.visualization.draw_geometries([partial_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame()])
        depth = np.array(Image.open(f'depth_test.png'), dtype=np.uint16)
        partial_pcd = RealSense.pointcloud(depth)

        # Remove outliers (just one outlier can lead to bad results)
        # partial_pcd, ind = partial_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0, print_progress=False)

        # Sample Point Cloud
        part = torch.FloatTensor(partial_pcd)  # Must be 1, 2024, 3
        # indices = fp_sampling(part.unsqueeze(0), 2024)  # TODO THIS IS SLOW
        # part = part[indices.long().squeeze()]  # TODO THIS IS SLOW
        part = part[torch.randperm(part.size()[0])]  # TODO THIS IS FAST BUT LESS ACCURATE (?)
        part = part[:2024]  # TODO THIS IS FAST BUT LESS ACCURATE (?)

        # Normalize Point Cloud as training time
        part = np.array(part)
        mean = np.mean(np.array(part), axis=0)
        part = np.array(part) - mean
        var = np.sqrt(np.max(np.sum(part ** 2, axis=1)))
        part = part / (var * 2)

        # part[..., -1] = -part[..., -1]  # TODO VERIFY (IS IT NORMAL THAT I NEED TO INVERT THIS?)
        partial_points = part

        # Reconstruct point cloud and find pose ########################################################################
        # Reconstruct partial point cloud
        complete_pc_aux, fast_weights = generator.reconstruct_point_cloud(partial_points)

        # Refine point cloud
        complete_pc_aux = generator.refine_point_cloud(complete_pc_aux, fast_weights, n=5, show_loss=False)

        # start = time.time()  # TODO REMOVE BOTTLENECK
        pc = PointCloud()
        pc.points = Vector3dVector(complete_pc_aux.squeeze(0).detach().cpu())
        complete_pc_aux = pc
        # print("TIME: {}".format(time.time() - start))

        # Find poses
        poses = generator.find_poses(complete_pc_aux, mult_res=1.5, n_points=1000, iterations=1000, debug=False)

        # Visualize results
        partial_pc_aux = PointCloud()
        partial_pc_aux.points = Vector3dVector(partial_points)  # NOTE: not a bottleneck because only 2048 points

        o3d.visualization.draw_geometries([partial_pc_aux, complete_pc_aux])  # TODO REMOVE
        test.run(partial_pc_aux, complete_pc_aux, poses)  # just pass partial points

        # If a key is pressed, break
        if msvcrt.kbhit():
            print(msvcrt.getch())
            break

        # TODO REMOVE DEBUG
        # pc = PointCloud()
        # pc.points = Vector3dVector(part)
        # o3d.visualization.draw_geometries([pc, o3d.geometry.TriangleMesh.create_coordinate_frame()])
        # inverted = np.array(pc.points)
        # inverted[..., 2] = -inverted[..., 2]
        # pc.points = Vector3dVector(inverted)
        # pc.scale(var*2, center=[0, 0, 0])
        # pc.translate(mean)
        # o3d.visualization.draw_geometries([pc, o3d.geometry.TriangleMesh.create_coordinate_frame(), partial_pcd])
        # TODO REMOVE DEBUG

