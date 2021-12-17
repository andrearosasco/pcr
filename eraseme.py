import numpy as np
from open3d.cpu.pybind.geometry import TriangleMesh, PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import Visualizer

vis = Visualizer()
vis.create_window()

coord = TriangleMesh().create_coordinate_frame(origin=np.random.rand(3))

# pc = PointCloud()
# pc.points = Vector3dVector(np.random.rand(100, 3))

# vis.add_geometry(pc)
vis.add_geometry(coord)

while True:

    # pc.clear()
    #
    # pc_ = PointCloud()
    # pc_.points = Vector3dVector(np.random.rand(100, 3))
    #
    # pc += pc_
    #
    # vis.update_geometry(pc)

    coord_ = TriangleMesh().create_coordinate_frame(origin=np.random.rand(3))

    coord.vertices = coord_.vertices
    coord.triangles = coord_.triangles

    vis.update_geometry(coord)

    vis.poll_events()
    vis.update_renderer()

