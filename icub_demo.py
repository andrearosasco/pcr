import yarp
import cv2
import numpy as np
import open3d as o3d
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector

from main import HyperNetwork
from configs.server_config import ModelConfig
from torch import cdist

from utils.misc import create_3d_grid

device = "cuda"


# TODO put presentation on teams, do video where I personally test model trained on AMI


def fp_sampling(points, num, starting_point=None):
    batch_size = points.shape[0]
    # If no starting_point is provided, the starting point is the first point of points
    if starting_point is None:
        starting_point = points[:, 0].unsqueeze(1)
    D = cdist(starting_point, points).squeeze(1)

    perm = torch.zeros((batch_size, num), dtype=torch.int32, device=points.device)
    ds = D
    for i in range(0, num):
        idx = torch.argmax(ds, dim=1)
        perm[:, i] = idx
        ds = torch.minimum(ds, cdist(points[torch.arange(batch_size), idx].unsqueeze(1), points).squeeze())

    return perm


class iCubGazebo:

    def __init__(self, rgb_port="/icubSim/cam/left/rgbImage:o", depth_port='/icubSim/cam/left/depthImage:o'):
        yarp.Network.init()

        # Create a port and connect it to the iCub simulator virtual camera
        self.rgb_port, self.depth_port = yarp.Port(), yarp.Port()
        self.rgb_port.open("/rgb-port")
        self.depth_port.open("/depth-port")
        yarp.Network.connect(rgb_port, "/rgb-port")
        yarp.Network.connect(depth_port, "/depth-port")

        self.rgb_array = np.zeros((240, 320, 3), dtype=np.uint8)
        self.rgb_image = yarp.ImageRgb()
        self.rgb_image.resize(320, 240)
        self.rgb_image.setExternal(self.rgb_array, self.rgb_array.shape[1], self.rgb_array.shape[0])

        self.depth_array = np.zeros((240, 320), dtype=np.float32)
        self.depth_image = yarp.ImageFloat()
        self.depth_image.resize(320, 240)
        self.depth_image.setExternal(self.depth_array, self.depth_array.shape[1], self.depth_array.shape[0])

    def read(self):
        self.rgb_port.read(self.rgb_image)
        self.depth_port.read(self.depth_image)

        return self.rgb_array[..., ::-1], self.depth_array


if __name__ == '__main__':
    icub = iCubGazebo()
    model = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
    model = model.to(device)
    model.eval()
    while True:
        rgb, depth = icub.read()
        cv2.imshow('RGB', rgb)  # TODO VISUALIZE DEBUG
        cv2.imshow('Depth', depth)  # TODO VISUALIZE DEBUG

        # Get only red part
        rgb_mask = rgb[..., 2] == 102  # Red is the last dimension
        rgb_mask = rgb_mask.astype(float) * 255
        cv2.imshow('Mask', rgb_mask)  # TODO VISUALIZE DEBUG

        # Get only depth of the box
        filtered_depth = np.where(rgb_mask, depth, 0.)
        filtered_depth_img = filtered_depth.astype(float) * 255
        cv2.imshow('Filtered Depth', filtered_depth_img)  # TODO VISUALIZE DEBUG

        # Convert depth image to Point Cloud
        fx = fy = 343.12110728152936
        cx = 160.0
        cy = 120.0
        intrinsics = o3d.camera.PinholeCameraIntrinsic(320, 240, fx, fy, cx, cy)
        partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(filtered_depth), intrinsics)
        # o3d.visualization.draw_geometries([pc])  # TODO VISUALIZE DEBUG

        # Sample Point Cloud
        partial = torch.FloatTensor(np.array(partial_pcd.points))  # Must be 1, 2024, 3
        print("Starting fps...")
        indices = fp_sampling(partial.unsqueeze(0), 2024)  # TODO THIS IS SLOW
        print("END")
        partial = partial[indices.long().squeeze()]

        # Normalize Point Cloud as training time
        partial = np.array(partial)
        mean = np.mean(np.array(partial), axis=0)
        partial = np.array(partial) - mean
        var = np.sqrt(np.max(np.sum(partial ** 2, axis=1)))
        partial = partial / (var * 2)

        partial[..., -1] = -partial[..., -1]  # TODO VERIFY

        partial_pcd = PointCloud()
        partial_pcd.points = Vector3dVector(partial)
        colors = np.array([0, 255, 0])[None, ...].repeat(partial.shape[0], axis=0)
        partial_pcd.colors = Vector3dVector(colors)

        # TODO START REMOVE DEBUG
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([partial_pcd])
        # TODO END REMOVE DEBUG

        # Inference
        partial = torch.FloatTensor(partial).unsqueeze(0).to(device)
        prediction = model(partial, step=0.01)  # TODO step SHOULD ME 0.01

        # Visualize
        prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()

        samples = create_3d_grid(batch_size=partial.shape[0], step=0.01)  # TODO we create grid two times...
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

        o3d.visualization.draw_geometries([pred_pc, part_pc])

        # continue

        # Create mesh from point cloud
        hull, _ = pred_pc.compute_convex_hull()
        # o3d.visualization.draw_geometries([hull])

        convex_pc = PointCloud()
        convex_pc.points = hull.vertices

        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(convex_pc)
        for alpha in [0.5]:
            print(f"alpha={alpha:.3f}")
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                convex_pc, alpha, tetra_mesh, pt_map)
            rec_mesh.compute_vertex_normals()
            print("REC:", rec_mesh.is_watertight())
            # print("ORIGINAL", mesh.is_watertight())
            # o3d.visualization.draw_geometries([rec_mesh, mesh.translate([1, 1, 1]), part_pc.translate([1, 1, 1])], mesh_show_back_face=True)
            o3d.visualization.draw_geometries([rec_mesh, part_pc], mesh_show_back_face=True)

        #  Create grasping
        rec_pc = rec_mesh.sample_points_uniformly(number_of_points=10000)
        rec_pc.estimate_normals()
        o3d.visualization.draw_geometries([rec_pc])

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
