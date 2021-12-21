import numpy as np
import open3d as o3d

from utils.misc import project_onto_plane, angle_between

try:
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    print("Open3d CUDA not found!")
    from open3d.cpu.pybind.geometry import PointCloud


# TODO NOTE
# In order to see the visualization correctly, one should stay in the coordinate frame, looking towards +z with +x
# facing towards -x and +y facing towards -y


class PoseGenerator:
    def __init__(self):
        pass

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

    @staticmethod
    def find_poses(pc, dist, n_points=10, iterations=100, debug=False):
        """
        Get a complete point cloud and return the a point cloud with good grasping spot
        Args:
            dist: distance of points in RANSAC algorithm to belong to plane
            iterations: number of iterations to do in segment plane
            n_points: number of points to use in each iteration of segment plane
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
            plane_model, ids = aux_pc.segment_plane(dist, n_points, iterations)  # TODO MODIFY
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
        c1, n1, c2, n2 = candidates[0][1], candidates[0][0], candidates[1][1], candidates[1][0]

        # TODO GENERATE POSE
        # Move center
        mean = np.array([0, 0, 0])
        var = 1
        c1 = c1 * var * 2  # De-normalize center
        c2 = c2 * var * 2  # De-normalize center
        c2[0] = -c2[0]  # Invert depth
        c2[0] = -c2[0]  # Invert depth
        c1 = c1 + mean
        c1 = c1 + mean

        # Create rotation matrix
        rotations = []
        for n in [n1, n2]:
            R_z = PoseGenerator.create_rotation_matrix(np.array([0, 0, 1]), n)

            ##########################
            # MAKE y AXE POINTS DOWN #
            ##########################
            y = (R_z @ np.array([0, 1, 0])) / np.linalg.norm(R_z @ np.array([0, 1, 0]))

            # Project y axis over the plane
            projected = project_onto_plane(np.array([0, 1, 0]), n)  # TODO CAREFull TO POINT DOWN
            projected = np.array(projected) / np.linalg.norm(projected)

            # Compute angle between projected y axe and actual y axe
            ort = np.cross(y, projected) / np.linalg.norm(np.cross(y, projected))
            sign = 1 if abs(np.sum(ort - n)) < 1e-8 else -1
            rotation_radians = angle_between(y, projected) * sign
            # print(np.degrees(rotation_radians))  # TODO remove debug

            # Rotate mesh
            C = np.array([[0, -n[2], n[1]],
                          [n[2], 0, -n[0]],
                          [-n[1], n[0], 0]])
            R_y = np.eye(3) + C * np.sin(rotation_radians) + C@C * (1 - np.cos(rotation_radians))

            # rotation_radians = angle_between(R_y @ y, projected) * sign  # TODO remove debug
            # print(np.degrees(rotation_radians))  # TODO remove debug

            rotations.append(R_y @ R_z)

        return c1, rotations[0], c2, rotations[1]
