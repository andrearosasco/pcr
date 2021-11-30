###########################
# NOT USE THE MESH AT ALL #
###########################
import time
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

        # Estimate normals
        pred_pc.estimate_normals()
        pred_pc.orient_normals_consistent_tangent_plane(10)

        return pred_pc

    def find_poses(self, pc):
        """
        Get a complete point cloud and return the a point cloud with good grasping spot
        Args:
            pc: Complete Point Cloud

        Returns:
            final_points: Point cloud of good grasping spots
        """
        # Run RANSAC for every face
        centers = []
        aux_pc = PointCloud(pc)
        for i in range(6):
            points = aux_pc.segment_plane(self.grid_res*4, 10, 100)  # TODO FINE TUNE THIS PARAMETERS
            points_list = np.array(points[1])
            plane_points = np.array(aux_pc.points)[points_list]

            centers.append(np.mean(plane_points, axis=0))

            aux_pc = aux_pc.select_by_index(points[1], invert=True)
            # o3d.visualization.draw_geometries([aux_pc])  # TODO VISUALIZE DEBUG

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


if __name__ == '__main__':

    res = 0.01

    # Setting up environment
    icub = iCubGazebo()
    model = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
    model = model.to(device)
    model.eval()

    # TODO START TRY TO PUT CLASS
    generator = FromPartialToPose(model, res)
    # TODO END START TRY

    # Main loop
    while True:

        # Get image
        rgb, depth = icub.read()

        # Get only red part
        rgb_mask = rgb[..., 2] == 102  # Red is the last dimension
        rgb_mask = rgb_mask.astype(float) * 255

        # Get only depth of the box
        filtered_depth = np.where(rgb_mask, depth, 0.)
        filtered_depth_img = filtered_depth.astype(float) * 255

        # TODO REMOVE DEBUG (VISUALIZE ICUB EYES OUTPUT)
        # cv2.imshow('RGB', rgb)  # TODO VISUALIZE DEBUG
        # cv2.imshow('Depth', depth)  # TODO VISUALIZE DEBUG
        # cv2.imshow('Mask', rgb_mask)  # TODO VISUALIZE DEBUG
        # cv2.imshow('Filtered Depth', filtered_depth_img)  # TODO VISUALIZE DEBUG

        # Convert depth image to Point Cloud
        fx = fy = 343.12110728152936
        cx = 160.0
        cy = 120.0
        intrinsics = o3d.camera.PinholeCameraIntrinsic(320, 240, fx, fy, cx, cy)
        partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(filtered_depth), intrinsics)
        # o3d.visualization.draw_geometries([pc])  # TODO VISUALIZE DEBUG

        # Sample Point Cloud
        partial = torch.FloatTensor(np.array(partial_pcd.points))  # Must be 1, 2024, 3
        start = time.time()  # TODO REMOVE DEBUG
        indices = fp_sampling(partial.unsqueeze(0), 2024)  # TODO THIS IS SLOW
        print("fps took: {}".format(time.time() - start))  # TODO REMOVE DEBUG
        partial = partial[indices.long().squeeze()]

        # Normalize Point Cloud as training time
        partial = np.array(partial)
        mean = np.mean(np.array(partial), axis=0)
        partial = np.array(partial) - mean
        var = np.sqrt(np.max(np.sum(partial ** 2, axis=1)))
        partial = partial / (var * 2)

        partial[..., -1] = -partial[..., -1]  # TODO VERIFY (IS IT NORMAL THAT I NEED TO INVERT THIS?)

        # TODO START TRY TO PUT CLASS
        complete = generator.reconstruct_point_cloud(partial)
        poses = generator.find_poses(complete)
        coords = generator.orient_poses(poses)
        o3d.visualization.draw_geometries(coords + [complete])
        continue
        # TODO END START TRY

        # TODO REMOVE DEBUG (VISUALIZE NORMALIZED PARTIAL POINT CLOUD)
        # partial_pcd = PointCloud()  # TODO VISUALIZE DEBUG
        # partial_pcd.points = Vector3dVector(partial)  # TODO VISUALIZE DEBUG
        # colors = np.array([0, 255, 0])[None, ...].repeat(partial.shape[0], axis=0)  # TODO VISUALIZE DEBUG
        # partial_pcd.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        # coord = o3d.geometry.TriangleMesh.create_coordinate_frame()  # TODO VISUALIZE DEBUG
        # o3d.visualization.draw_geometries([partial_pcd])  # TODO VISUALIZE DEBUG

        # Inference
        partial = torch.FloatTensor(partial).unsqueeze(0).to(device)
        prediction = model(partial, step=res)  # TODO step SHOULD BE 0.01

        # Get the selected point on the grid
        prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()
        samples = create_3d_grid(batch_size=partial.shape[0], step=res)  # TODO we create grid two times...
        samples = samples.squeeze(0).detach().cpu().numpy()
        selected = samples[prediction > 0.5]
        pred_pc = PointCloud()
        pred_pc.points = Vector3dVector(selected)
        pred_pc.estimate_normals()
        pred_pc.orient_normals_consistent_tangent_plane(10)

        # TODO REMOVE DEBUG (VISUALIZE PARTIAL POINT CLOUD AND PREDICTED POINT CLOUD)
        # partial = partial.squeeze(0).detach().cpu().numpy()  # TODO VISUALIZE DEBUG
        # part_pc = PointCloud()  # TODO VISUALIZE DEBUG
        # part_pc.points = Vector3dVector(partial)  # TODO VISUALIZE DEBUG
        # colors = np.array([0, 255, 0])[None, ...].repeat(partial.shape[0], axis=0)  # TODO VISUALIZE DEBUG
        # part_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        # colors = np.array([0, 0, 255])[None, ...].repeat(selected.shape[0], axis=0)  # TODO VISUALIZE DEBUG
        # pred_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        # o3d.visualization.draw_geometries([pred_pc, part_pc])  # TODO VISUALIZE DEBUG

        # Run RANSAC for every face
        centers = []
        aux_pc = PointCloud(pred_pc)
        for i in range(6):
            points = aux_pc.segment_plane(res*4, 10, 100)  # TODO FINE TUNE THIS PARAMETERS
            points_list = np.array(points[1])
            plane_points = np.array(aux_pc.points)[points_list]

            centers.append(np.mean(plane_points, axis=0))

            aux_pc = aux_pc.select_by_index(points[1], invert=True)
            # o3d.visualization.draw_geometries([aux_pc])  # TODO VISUALIZE DEBUG

        # TODO VISUALIZE DEBUG (VISUALIZE RECONSTRUCTED MESH AND FACE CENTERS)
        # centers_pc = PointCloud()  # TODO VISUALIZE DEBUG
        # centers_pc.points = Vector3dVector(np.array(centers))  # TODO VISUALIZE DEBUG
        # colors = np.array([255, 0, 0])[None, ...].repeat(len(centers_pc.points), axis=0)  # TODO VISUALIZE DEBUG
        # centers_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        # o3d.visualization.draw_geometries([rec_mesh, centers_pc], mesh_show_back_face=True)  # TODO VISUALIZE DEBUG

        # Get closest point from sampled point cloud for every center
        true_centers = []
        sampled_mesh = torch.FloatTensor(np.array(pred_pc.points))
        centers = torch.FloatTensor(np.array(centers))
        for c in centers:
            c = c.unsqueeze(0)

            # TODO VISUALIZE DEBUG
            # c_pc = PointCloud()  # TODO VISUALIZE DEBUG
            # c_pc.points = Vector3dVector(np.array(c))  # TODO VISUALIZE DEBUG
            # colors = np.array([255, 0, 0])[None, ...].repeat(len(c), axis=0)  # TODO VISUALIZE DEBUG
            # c_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
            # o3d.visualization.draw_geometries([rec_pc, c_pc])  # TODO VISUALIZE DEBUG
            # TODO END VISUALIZE DEBUG

            dist = sampled_mesh - c
            dist = torch.square(dist[..., 0]) + torch.square(dist[..., 1]) + torch.square(dist[..., 2])
            true_centers.append(torch.argmin(dist).numpy())

        true_centers = np.array(true_centers).squeeze()

        # TODO VISUALIZE DEBUG (SHOWS RECONSTRUCTED MESH AND REAL CENTERS)
        final_points = pred_pc.select_by_index(true_centers)

        pred_pc.normals = Vector3dVector([])  # TODO VISUALIZE MEGA DEBUG

        geometries = [pred_pc, final_points]
        for c, normal in zip(np.array(final_points.points), np.array(final_points.normals)):
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=c, size=0.1)
            normal = normal/np.linalg.norm(normal)
            R = create_rotation_matrix(np.array([0, 0, 1]), normal)
            coord.rotate(R, center=c)
            geometries.append(coord)

        o3d.visualization.draw_geometries(geometries)

        # true_pc = PointCloud()  # TODO VISUALIZE DEBUG
        # true_pc.points = Vector3dVector(true_centers)
        # colors = np.array([255, 0, 0])[None, ...].repeat(len(true_centers), axis=0)  # TODO VISUALIZE DEBUG
        # true_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        # coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([true_pc, coord], mesh_show_back_face=True)  # TODO VISUALIZE DEBUG

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
