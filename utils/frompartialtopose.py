import time
import random
import yarp
import numpy as np
import open3d as o3d
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import Visualizer
from torch import cdist
from configs.server_config import ModelConfig
from main import HyperNetwork
from utils.misc import create_3d_grid


device = "cuda"


class GenPose:
    def __init__(self, device="cuda", res=0.01):

        # Random seed
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))

        # Class parameters
        self.device = device
        self.res = res
        # Pose generator
        model = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
        model = model.to(device)
        model.eval()
        self.generator = FromPartialToPose(model, res)

        # Set up visualizer
        self.vis = Visualizer()
        self.vis.create_window()
        # Complete point cloud
        self.complete_pc = PointCloud()
        self.complete_pc.points = Vector3dVector(np.random.randn(2348, 3))
        self.vis.add_geometry(self.complete_pc)
        # Partial point cloud
        self.partial_pc = PointCloud()
        self.partial_pc.points = Vector3dVector(np.random.randn(2024, 3))
        self.vis.add_geometry(self.partial_pc)
        # Camera TODO ROTATE
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 1]))
        # Coords
        self.best1_rot = None
        self.best2_rot = None
        self.best1_mesh = None
        self.best2_mesh = None
        self.coords_mesh = None
        self.coords_rot = None

    def reset_coords(self):
        self.vis.remove_geometry(self.best1_mesh)
        self.vis.remove_geometry(self.best2_mesh)
        if self.coords_mesh is not None:
            for coord in self.coords_mesh:
                self.vis.remove_geometry(coord)

        self.best1_rot = np.array([0, 0, 1])
        self.best2_rot = np.array([0, 0, 1])
        self.best1_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.best2_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.vis.add_geometry(self.best1_mesh)
        self.vis.add_geometry(self.best2_mesh)

        self.coords_mesh = []
        self.coords_rot = []
        for _ in range(6):
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.coords_mesh.append(coord)
            self.coords_rot.append(np.array([0, 0, 1]))
            self.vis.add_geometry(coord)

    def run(self, partial_points):

        partial_points = np.array(partial_points)  # From list to array
        partial_pc_aux = PointCloud()
        partial_pc_aux.points = Vector3dVector(partial_points)

        # Reconstruct partial point cloud
        # start = time.time()
        complete_pc_aux = self.generator.reconstruct_point_cloud(partial_points)
        # print("Reconstruct: {}".format(time.time() - start))

        # Find poses
        poses = self.generator.find_poses(complete_pc_aux, mult_res=1.5, n_points=100, iterations=1000, debug=False)

        # Orient poses
        new_coords_rot = []
        highest_value = 0
        highest_id = -1
        lowest_value = 0
        lowest_id = -1
        i = 0
        for c, normal, coord_rot, coord_mesh in zip(np.array(poses.points), np.array(poses.normals),
                                                    self.coords_rot, self.coords_mesh):
            coord_mesh.translate(c, relative=False)
            normal = normal / np.linalg.norm(normal)
            R = FromPartialToPose.create_rotation_matrix(coord_rot, normal)
            coord_mesh.rotate(R, center=c)
            self.vis.update_geometry(coord_mesh)
            new_coord_rot = (R @ coord_rot) / np.linalg.norm(R @ coord_rot)
            new_coords_rot.append(new_coord_rot)

            if normal[0] > highest_value:
                highest_value = normal[0]
                highest_id = i
            if normal[0] < lowest_value:
                lowest_value = normal[0]
                lowest_id = i

            i += 1
        self.coords_rot = new_coords_rot

        # Update partial point cloud in visualizer
        self.partial_pc.clear()
        self.partial_pc += partial_pc_aux
        colors = np.array([0, 255, 0])[None, ...].repeat(len(self.partial_pc.points), axis=0)
        self.partial_pc.colors = Vector3dVector(colors)
        self.vis.update_geometry(self.partial_pc)

        # Update complete point cloud in visualizer
        self.complete_pc.clear()
        self.complete_pc += complete_pc_aux
        # colors = np.array([255, 0, 0])[None, ...].repeat(len(self.complete_pc.points), axis=0)
        # self.complete_pc.colors = Vector3dVector(colors)
        self.vis.update_geometry(self.complete_pc)

        # Update best points
        self.best1_mesh.translate(np.array(poses.points)[highest_id], relative=False)
        self.best2_mesh.translate(np.array(poses.points)[lowest_id], relative=False)

        R1 = FromPartialToPose.create_rotation_matrix(self.best1_rot, self.coords_rot[highest_id])
        self.best1_mesh.rotate(R1)
        self.best1_rot = R1 @ self.best1_rot
        self.best1_rot = self.best1_rot / np.linalg.norm(self.best1_rot)

        R2 = FromPartialToPose.create_rotation_matrix(self.best2_rot, self.coords_rot[lowest_id])
        self.best2_mesh.rotate(R2)
        self.best2_rot = R2 @ self.best2_rot
        self.best2_rot = self.best2_rot / np.linalg.norm(self.best2_rot)

        self.vis.update_geometry(self.best1_mesh)
        self.vis.update_geometry(self.best2_mesh)

        # Update visualizer
        self.vis.poll_events()
        self.vis.update_renderer()




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


class FromPartialToPose:
    def __init__(self, model, grid_res):
        self.model = model
        self.grid_res = grid_res

    @staticmethod
    def create_rotation_matrix(a, b):
        """
        Creates the rotation matrix such that b = R @ a
        from https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        Args:
            a: np.array (,3), unit vector already normalized
            b: np.array (,3), unit vector already normalized

        Returns:
            R: np.array(3, 3), transformation matrix
        """
        v = np.cross(a, b)

        c = np.dot(a, b)

        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])

        R = np.eye(3) + vx + np.dot(vx, vx) * (1 / (1 + c))
        return R

    def reconstruct_point_cloud(self, partial):
        """
        Given a partial point cloud, it reconstructs it
        Args:
            partial: np.array(N, 3)

        Returns:
            complete: o3d.Geometry.PointCloud with estimated normals
        """
        # Inference
        partial = torch.FloatTensor(partial).unsqueeze(0).to(device)
        prediction = self.model(partial, step=self.grid_res)  # TODO step SHOULD BE 0.01

        # Get the selected point on the grid
        prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()
        samples = create_3d_grid(batch_size=partial.shape[0], step=self.grid_res)  # TODO we create grid two times...
        samples = samples.squeeze(0).detach().cpu().numpy()
        selected = samples[prediction > 0.5]
        pred_pc = PointCloud()
        pred_pc.points = Vector3dVector(selected)

        #TODO START EXPERIMENT
        pred_pc = pred_pc.random_down_sample(0.1)
        # start = time.time()
        # pred_pc.estimate_normals()
        # pred_pc.orient_normals_consistent_tangent_plane(10)
        # print("EXPERIMENTING NORMALs: {}".format(time.time() - start))
        # o3d.visualization.draw_geometries([pred_pc])
        #TODO END EXPERIMENT

        # Estimate normals
        # start = time.time()
        pred_pc.estimate_normals()
        pred_pc.orient_normals_consistent_tangent_plane(3)
        # print("Estimate normals: {}".format(time.time() - start))

        return pred_pc

    def find_poses(self, pc, mult_res=4, n_points=10, iterations=100, debug=False):
        """
        Get a complete point cloud and return the a point cloud with good grasping spot
        Args:
            iterations: number of iterations to do in segment plane
            n_points: number of points to use in each iteration of segment plane
            mult_res: multiplier of grid resolution to use in segment plane
            pc: Complete Point Cloud

        Returns:
            final_points: Point cloud of good grasping spots
        """
        # Run RANSAC for every face
        centers = []
        aux_pc = PointCloud(pc)
        for i in range(6):

            # There are not enough plane in the reconstructed shape
            if len(aux_pc.points) < n_points:
                for _ in range(6 - i):
                    centers.append(np.array([0, 0, 0]))
                print("Segment plane does not have enough points")
                break

            points = aux_pc.segment_plane(self.grid_res*mult_res, n_points, iterations)
            points_list = np.array(points[1])
            plane_points = np.array(aux_pc.points)[points_list]

            centers.append(np.mean(plane_points, axis=0))

            aux_pc = aux_pc.select_by_index(points[1], invert=True)
            if debug:
                o3d.visualization.draw_geometries([aux_pc])

        # Get closest point from sampled point cloud for every center
        true_centers = []
        sampled_mesh = torch.FloatTensor(np.array(pc.points))
        centers = torch.FloatTensor(np.array(centers))
        for c in centers:
            c = c.unsqueeze(0)
            dist = sampled_mesh - c
            dist = torch.square(dist[..., 0]) + torch.square(dist[..., 1]) + torch.square(dist[..., 2])
            true_centers.append(torch.argmin(dist).numpy())

        true_centers = np.array(true_centers).squeeze()

        final_points = pc.select_by_index(true_centers)
        return final_points

    def orient_poses(self, poses):
        """
        Construct coordinate frame accordingly to the estimated poses and its normals
        Args:
            poses:

        Returns:

        """
        geometries = []
        for c, normal in zip(np.array(poses.points), np.array(poses.normals)):
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=c, size=0.1)
            normal = normal/np.linalg.norm(normal)
            R = self.create_rotation_matrix(np.array([0, 0, 1]), normal)
            coord.rotate(R, center=c)
            geometries.append(coord)

        return geometries
