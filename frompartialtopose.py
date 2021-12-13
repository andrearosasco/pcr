import copy
import numpy as np
import open3d as o3d
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from torch import cdist
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
import time


# TODO NOTE
# In order to see the visualization correctly, one should stay in the coordinate frame, looking towards +z with +x
# facing towards -x and +y facing towards -y


class FromPartialToPose:
    def __init__(self, model, grid_res, device="cuda"):
        self.model = model
        self.grid_res = grid_res
        self.device = device

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

    def refine_point_cloud(self, complete_pc, fast_weights, partial_pc, n=10, show_loss=False):
        """
        Uses adversarial attack to refine the generated complete point cloud
        Args:
            complete_pc: torch.Tensor(N, 3)
            fast_weights: List ( List ( Tensor ) )
            partial_pc: torch.Tensor(N, 3)
            n: Int, number of adversarial steps
        Returns:
            pc: PointCloud
        """
        complete_pc = complete_pc.unsqueeze(0)  # add batch dimension
        complete_pc.requires_grad = True
        partial_pc = partial_pc.unsqueeze(0)  # add batch dimension
        complete_pc_0 = copy.deepcopy(complete_pc)

        loss_function = BCEWithLogitsLoss(reduction='mean')
        optim = SGD([complete_pc], lr=0.5, momentum=0.9)

        c1, c2, c3 = 1, 0, 0  # 1, 0, 0  1, 1e3, 0 # 0, 1e4, 5e2
        for step in range(n):
            results = self.model.sdf(complete_pc, fast_weights)

            gt = torch.ones_like(results[..., 0], dtype=torch.float32)
            gt[:, :] = 1
            loss1 = c1 * loss_function(results[..., 0], gt)
            loss2 = c2 * torch.mean((complete_pc - complete_pc_0) ** 2)
            loss3 = c3 * torch.mean(cdist(complete_pc, partial_pc).sort(dim=2)[0][:, :,
                                    0])  # it works but it would be nicer to do the opposite
            loss_value = loss1 + loss2 + loss3

            self.model.zero_grad()
            optim.zero_grad()
            loss_value.backward(inputs=[complete_pc])
            optim.step()
            if show_loss:
                print('Loss ', loss_value.item())

        pc = PointCloud()
        pc.points = Vector3dVector(complete_pc.squeeze(0).detach().cpu())
        return pc

    @staticmethod
    def estimate_normals(pc, n=5):
        """
        It estimates the normal of the point cloud. This function could be computationally expensive
        Args:
            pc: PointCloud
            n: Number of points for the knn algorithm to build the graph which estimates the tangents
            show_time: If True, this function outputs the number of time used to do the function

        Returns:

        """
        start = time.time()
        pc.estimate_normals()
        pc.orient_normals_consistent_tangent_plane(n)
        return pc

    def reconstruct_point_cloud(self, partial):
        """
        Given a partial point cloud, it reconstructs it
        Args:
            partial: np.array(N, 3)

        Returns:
            selected: Torch.Tensor(N, 3)
            fast_weights: List( List( Torch.Tensor ) )
        """
        # Inference
        partial = torch.FloatTensor(partial).unsqueeze(0).to(self.device)
        prediction, fast_weights, samples = self.model(partial, step=self.grid_res)  # TODO step SHOULD BE 0.01

        # Get the selected point on the grid
        prediction = prediction.squeeze(0).squeeze(-1).detach()
        samples = samples.squeeze(0).detach()
        selected = samples[prediction > 0.5]

        return selected, fast_weights

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
