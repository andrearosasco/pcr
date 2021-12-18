import numpy as np
from open3d.cpu.pybind.geometry import TriangleMesh, PointCloud, LineSet
from open3d.cpu.pybind.utility import Vector3dVector, Vector2iVector
from open3d.cpu.pybind.visualization import Visualizer
import open3d as o3d

from utils.pose_visualizer import project_onto_plane, angle_between


def create_vector(v, c):
    line = LineSet()
    line.points = Vector3dVector(np.array([[0, 0, 0],
                                           v]))
    line.lines = Vector2iVector(np.array([[0, 1]]))
    line.colors = Vector3dVector(np.array([c]))
    return line


plane = np.array([1, 1, 1])
plane = plane / np.linalg.norm(plane)
plane_c = np.array([0, 255, 255])

target = np.array([1, -1, 1])
target = project_onto_plane(target, plane)
target = target / np.linalg.norm(target)
target_c = np.array([0, 255, 0])

to_proj = np.array([5, 1, 1])
to_proj = to_proj / np.linalg.norm(to_proj)
to_proj_c = np.array([0, 0, 255])

projected = project_onto_plane(to_proj, plane)

projected = projected / np.linalg.norm(projected)
projected_c = np.array([0, 0, 0])

rotation_radians = angle_between(projected, target)
n = np.cross(projected, target) / np.linalg.norm(np.cross(projected, target))
sign = 1 if abs(np.sum(n - plane)) < 1e-8 else -1
print(np.degrees(rotation_radians))
print(sign)
print(n - plane)
rotation_radians = rotation_radians * sign

C = np.array([[0, -plane[2], plane[1]],
              [plane[2], 0, -plane[0]],
              [-plane[1], plane[0], 0]])
R = np.eye(3) + C * np.sin(rotation_radians) + C @ C * (1 - np.cos(rotation_radians))

result = R @ projected
result = result / np.linalg.norm(result)
result_c = np.array([255, 0, 0])

rotation_radians = angle_between(result, target)
print(np.degrees(rotation_radians))

o3d.visualization.draw_geometries([
                                   create_vector(plane, plane_c),
                                   create_vector(target, target_c),
                                   create_vector(to_proj, to_proj_c),
                                   create_vector(projected, projected_c),
                                   create_vector(result, result_c),
                                   TriangleMesh.create_coordinate_frame()])

exit()

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
