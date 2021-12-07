import open3d as o3d
import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import Visualizer

pc = PointCloud()
pc.points = Vector3dVector(np.random.randn(100, 3))

vis = Visualizer()
vis.create_window()
vis.add_geometry(pc)

while True:
    pc.points = Vector3dVector(np.random.randn(100, 3))

    vis.update_geometry(pc)

    vis.poll_events()
    vis.update_renderer()

