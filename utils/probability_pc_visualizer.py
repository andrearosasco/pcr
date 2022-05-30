import open3d as o3d
import numpy as np
import math


def visualize_probability_pc(pc):
    """
    :param pc: array with dimension (n_points, 4) where pc[i] = (x, y, x, p) and p is probability [0, 1]
    :return: None
    """
    points = pc[:, :3]
    probabilities = pc[:, 3]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    aux = np.zeros((len(probabilities), 3), dtype=np.float)
    aux[:, 0] = probabilities
    pc.colors = o3d.utility.Vector3dVector(aux)
    o3d.visualization.draw_geometries([pc])


if __name__ == "__main__":
    filename = "synthetic_dataset/ycb/001_chips_can/clouds/merged_cloud.ply"
    pcd = o3d.io.read_point_cloud(filename, format='ply')
    points = np.array(pcd.points)
    probability = np.array([math.dist(x, (0, 0, 0)) for x in points])
    maximum = max(probability)
    probability = probability / maximum
    probability = probability[:, None]
    points = np.concatenate((points, probability), -1)
    visualize_probability_pc(points)
