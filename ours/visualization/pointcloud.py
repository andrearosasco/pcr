import numpy as np
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector
from open3d.open3d.visualization import draw_geometries


def draw_point_cloud(x):
    x = x.cpu().squeeze()
    pc = PointCloud()
    pc.points = Vector3dVector(np.array(x))
    draw_geometries([pc])


crop_ratio = {
    'easy': 1/4,
    'median': 1/2,
    'hard': 3/4
}