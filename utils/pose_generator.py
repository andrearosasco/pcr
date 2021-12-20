import numpy as np
import open3d as o3d
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD

try:
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    print("Open3d CUDA not found!")
    from open3d.cpu.pybind.geometry import PointCloud


# TODO NOTE
# In order to see the visualization correctly, one should stay in the coordinate frame, looking towards +z with +x
# facing towards -x and +y facing towards -y


class PoseGenerator:
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

    # def refine_point_cloud(self, complete_pc, fast_weights, n=10, show_loss=False):
    #     """
    #     Uses adversarial attack to refine the generated complete point cloud
    #     Args:
    #         show_loss: if True, the loss is printed at each iteration (useful to find correct value)
    #         complete_pc: torch.Tensor(N, 3)
    #         fast_weights: List ( List ( Tensor ) )
    #         n: Int, number of adversarial steps
    #     Returns:
    #         pc: PointCloud
    #     """
    #     complete_pc = complete_pc.unsqueeze(0)  # add batch dimension
    #     complete_pc.requires_grad = True
    #
    #     loss_function = BCEWithLogitsLoss(reduction='mean')
    #     optim = SGD([complete_pc], lr=0.5, momentum=0.9)
    #
    #     for step in range(n):
    #         results = self.model.sdf(complete_pc, fast_weights)
    #
    #         gt = torch.ones_like(results[..., 0], dtype=torch.float32)
    #         gt[:, :] = 1
    #         loss_value = loss_function(results[..., 0], gt)
    #
    #         self.model.zero_grad()
    #         optim.zero_grad()
    #         loss_value.backward(inputs=[complete_pc])
    #         optim.step()
    #         # if show_loss:
    #         #     print('Loss ', loss_value.item())
    #     return complete_pc

    # def reconstruct_point_cloud(self, partial):
    #     """
    #     Given a partial point cloud, it reconstructs it
    #     Args:
    #         partial: np.array(N, 3)
    #
    #     Returns:
    #         selected: Torch.Tensor(N, 3)
    #         fast_weights: List( List( Torch.Tensor ) )
    #     """
    #     # Inference
    #     partial = torch.FloatTensor(partial).unsqueeze(0).to(self.device)
    #     prediction, fast_weights, samples = self.model(partial, step=self.grid_res)
    #
    #     # Get the selected point on the grid
    #     prediction = prediction.squeeze(0).squeeze(-1).detach()
    #     samples = samples.squeeze(0).detach()
    #     selected = samples[prediction > 0.5]
    #
    #     return selected, fast_weights

    def find_poses(self, pc, mult_res=4, n_points=10, iterations=100, debug=False):
        """
        Get a complete point cloud and return the a point cloud with good grasping spot
        Args:
            iterations: number of iterations to do in segment plane
            n_points: number of points to use in each iteration of segment plane
            mult_res: multiplier of grid resolution to use in segment plane
            pc: Complete Point Cloud

        Returns:
            poses: first best center, first best normal, second best center, second best normal
        """
        # Run RANSAC for every face
        centers = []
        aux_pc = PointCloud(pc)
        candidates = []

        for i in range(6):
            # There are not enough plane in the reconstructed shape
            if len(aux_pc.points) < n_points:
                for _ in range(6 - i):
                    centers.append(np.array([0, 0, 0]))
                print("Segment plane does not have enough points")
                break
            plane_model, ids = aux_pc.segment_plane(self.grid_res*mult_res, n_points, iterations)  # TODO MODIFY
            points_list = np.array(ids)
            plane_points = np.array(aux_pc.points)[points_list]

            # Normalize plane normals
            plane_model = plane_model[:3]
            plane_model = plane_model / np.linalg.norm(plane_model)

            candidates.append((plane_model, np.mean(plane_points, axis=0)))

            aux_pc = aux_pc.select_by_index(ids, invert=True)
            if debug:
                o3d.visualization.draw_geometries([aux_pc])

        candidates = sorted(candidates, reverse=True, key=lambda x: abs(x[0][0]))
        return candidates[0][1], candidates[0][0], candidates[1][1], candidates[1][0]

    def orient_poses(self):
        pass
        # new_vert_rot = (R @ np.array([0, 1, 0])) / np.linalg.norm(R @ np.array([0, 1, 0]))
        #
        # # Project y axis over the plane
        # projected = project_onto_plane(np.array([0, 1, 0]), normal)  # TODO CAREFull TO POINT DOWN
        # projected = np.array(projected) / np.linalg.norm(projected)
        #
        # # Compute angle between projected y axe and actual y axe
        # n = np.cross(new_vert_rot, projected) / np.linalg.norm(np.cross(new_vert_rot, projected))
        # sign = 1 if abs(np.sum(n - normal)) < 1e-8 else -1
        # rotation_radians = angle_between(new_vert_rot, projected) * sign
        # # print(np.degrees(rotation_radians))
