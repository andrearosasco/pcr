import time
import random
import open3d as o3d
import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import Visualizer
from frompartialtopose import FromPartialToPose
from main import HyperNetwork
from configs.server_config import ModelConfig


class PoseVisualizer:
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
        self.generator = FromPartialToPose(model, res, device)

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
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
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

    def run(self, partial_points, mean=(0, 0, 0), var=1, depth_pc=None):

        partial_points = np.array(partial_points)  # From list to array
        partial_pc_aux = PointCloud()
        partial_pc_aux.points = Vector3dVector(partial_points)

        # Reconstruct partial point cloud
        start = time.time()
        complete_pc_aux, fast_weights = self.generator.reconstruct_point_cloud(partial_points)
        print("Reconstruct: {}".format(time.time() - start))

        start = time.time()
        complete_pc_aux = self.generator.refine_point_cloud(complete_pc_aux, fast_weights, partial_points, n=0,
                                                            show_loss=False)
        print("Refine: {}".format(time.time() - start))

        complete_pc_aux = complete_pc_aux.voxel_down_sample(self.res*2)  # Reduce dimension of the point cloud

        start = time.time()
        complete_pc_aux = self.generator.estimate_normals(complete_pc_aux)
        print("Estimate normals: {}".format(time.time() - start))

        start = time.time()
        poses = self.generator.find_poses(complete_pc_aux, mult_res=1.5, n_points=100, iterations=1000, debug=False)
        print("Find poses: {}".format(time.time() - start))

        # Orient poses
        new_coords_rot = []
        highest_value = 0
        highest_id = -1
        lowest_value = 0
        lowest_id = -1
        i = 0
        for c, normal, coord_rot, coord_mesh in zip(np.array(poses.points), np.array(poses.normals),
                                                    self.coords_rot, self.coords_mesh):

            c[2] = -c[2]  # Invert depth
            normal[2] = -normal[2]  # Invert normal in depth dimension
            c = c*var*2  # De-normalize center

            coord_mesh.translate(c, relative=False)
            normal = normal / np.linalg.norm(normal)
            R = FromPartialToPose.create_rotation_matrix(coord_rot, normal)
            coord_mesh.rotate(R, center=c)

            coord_mesh.translate(mean, relative=True)  # Translate coord as the point cloud

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
        # NOTE: this point cloud (after the de normalization) overlaps with the point cloud reconstructed from the depth
        self.partial_pc.clear()
        self.partial_pc += partial_pc_aux
        colors = np.array([0, 255, 0])[None, ...].repeat(len(self.partial_pc.points), axis=0)
        self.partial_pc.colors = Vector3dVector(colors)
        # Invert x axis to plot like the pc obtained from depth
        inverted = np.array(self.partial_pc.points)
        inverted[..., 2] = -inverted[..., 2]
        self.partial_pc.points = Vector3dVector(inverted)
        # De-normalize
        self.partial_pc.scale(var*2, center=[0, 0, 0])
        self.partial_pc.translate(mean)
        self.vis.update_geometry(self.partial_pc)

        # Update complete point cloud in visualizer
        self.complete_pc.clear()
        self.complete_pc += complete_pc_aux
        colors = np.array([255, 0, 0])[None, ...].repeat(len(self.complete_pc.points), axis=0)
        self.complete_pc.colors = Vector3dVector(colors)
        # Invert x axis to plot like the pc obtained from the depth
        inverted = np.array(self.complete_pc.points)
        inverted[..., 2] = -inverted[..., 2]
        self.complete_pc.points = Vector3dVector(inverted)
        # De-normalize
        self.complete_pc.scale(var*2, center=[0, 0, 0])
        self.complete_pc.translate(mean)
        self.vis.update_geometry(self.complete_pc)

        # Update best points
        c = np.array(poses.points)[highest_id]
        c[2] = -c[2]  # Invert depth
        c = c * var * 2  # De-normalize center
        self.best1_mesh.translate(c, relative=False)
        self.best1_mesh.translate(mean, relative=True)

        c = np.array(poses.points)[lowest_id]
        c[2] = -c[2]  # Invert depth
        c = c * var * 2  # De-normalize center
        self.best2_mesh.translate(c, relative=False)
        self.best2_mesh.translate(mean, relative=True)

        inverted_pose = self.coords_rot[highest_id]  # TODO VERIFY
        inverted_pose = -inverted_pose  # TODO VERIFY
        R1 = FromPartialToPose.create_rotation_matrix(self.best1_rot, inverted_pose)  # TODO VERIFY
        self.best1_mesh.rotate(R1)
        self.best1_rot = R1 @ self.best1_rot
        self.best1_rot = self.best1_rot / np.linalg.norm(self.best1_rot)
        # TODO EXPERIMENT
        R1 = FromPartialToPose.create_rotation_matrix(self.best1_rot, inverted_pose)  # TODO VERIFY
        # TODO END EXPERIMENT

        R2 = FromPartialToPose.create_rotation_matrix(self.best2_rot, self.coords_rot[lowest_id])
        self.best2_mesh.rotate(R2)
        self.best2_rot = R2 @ self.best2_rot
        self.best2_rot = self.best2_rot / np.linalg.norm(self.best2_rot)

        self.vis.update_geometry(self.best1_mesh)
        self.vis.update_geometry(self.best2_mesh)

        # Update visualizer
        self.vis.poll_events()
        self.vis.update_renderer()

        # Return results
        xyz_left = np.array(poses.points)[highest_id]
        xyz_left[2] = -xyz_left[2]  # Invert depth
        xyz_left = xyz_left * var * 2  # De-normalize center

        xyz_right = np.array(poses.points)[lowest_id]
        xyz_right[2] = -xyz_right[2]  # Invert depth
        xyz_right = xyz_right * var * 2  # De-normalize center

        # self.complete_pc.estimate_normals()  # TODO REMOVE
        # self.complete_pc.orient_normals_consistent_tangent_plane(5)  # TODO REMOVE
        # #
        # g = self.coords_mesh + [self.best1_mesh, self.best2_mesh, self.partial_pc, self.complete_pc,
        #                         o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]
        # o3d.visualization.draw_geometries(g)

        return xyz_left, self.coords_rot[highest_id], xyz_right, self.coords_rot[lowest_id]

        # Visualize point cloud vision from robot's perspective
        # rgb = np.array(self.vis.capture_screen_float_buffer())
        # rgb = cv2.resize(rgb, (320, 240))
        # cv2.imshow("PC", rgb)

        # TODO REMOVE DEBUG
        # g = self.coords_mesh + [self.best1_mesh, self.best2_mesh, self.partial_pc, self.complete_pc,
        #                         o3d.geometry.TriangleMesh.create_coordinate_frame()]
        # o3d.visualization.draw_geometries(g)

        # o3d.visualization.draw_geometries([depth_pc, self.partial_pc])

        # o3d.visualization.draw_geometries([self.complete_pc, depth_pc, o3d.geometry.TriangleMesh.create_coordinate_frame()])