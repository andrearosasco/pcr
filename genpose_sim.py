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
        start = time.time()
        pred_pc.estimate_normals()
        pred_pc.orient_normals_consistent_tangent_plane(3)
        print("Estimate normals: {}".format(time.time() - start))

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


if __name__ == '__main__':

    res = 0.01

    # Setting up environment
    icub = iCubGazebo()
    m = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
    m = m.to(device)
    m.eval()

    # TODO START TRY TO PUT CLASS
    generator = FromPartialToPose(m, res)
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
        # key = cv2.waitKey(1) & 0xFF  # TODO VISUALIZE DEBUG
        # if key == ord("q"):  # TODO VISUALIZE DEBUG
        #     break  # TODO VISUALIZE DEBUG

        # Convert depth image to Point Cloud
        fx = fy = 343.12110728152936
        cx = 160.0
        cy = 120.0
        intrinsics = o3d.camera.PinholeCameraIntrinsic(320, 240, fx, fy, cx, cy)
        partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(filtered_depth), intrinsics)
        # o3d.visualization.draw_geometries([pc])  # TODO VISUALIZE DEBUG

        # Sample Point Cloud
        part = torch.FloatTensor(np.array(partial_pcd.points))  # Must be 1, 2024, 3
        start = time.time()  # TODO REMOVE DEBUG
        indices = fp_sampling(part.unsqueeze(0), 2024)  # TODO THIS IS SLOW
        print("fps took: {}".format(time.time() - start))  # TODO REMOVE DEBUG
        part = part[indices.long().squeeze()]

        # Normalize Point Cloud as training time
        part = np.array(part)
        mean = np.mean(np.array(part), axis=0)
        part = np.array(part) - mean
        var = np.sqrt(np.max(np.sum(part ** 2, axis=1)))
        part = part / (var * 2)

        part[..., -1] = -part[..., -1]  # TODO VERIFY (IS IT NORMAL THAT I NEED TO INVERT THIS?)

        complete = generator.reconstruct_point_cloud(part)
        p = generator.find_poses(complete, mult_res=4, n_points=10, iterations=100)
        coords = generator.orient_poses(p)
        o3d.visualization.draw_geometries(coords + [complete])
