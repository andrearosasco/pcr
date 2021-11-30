import open3d as o3d
import numpy as np
from open3d.cpu.pybind.utility import Vector3dVector, Vector2iVector


def create_rotation_matrix(a, b):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(a, b)

    c = np.dot(a, b)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + np.dot(vx, vx) * (1 / (1 + c))
    return R


l = 1

source = [0, 0, l]
target = [-l, -l, -l]


a = np.array(source)/np.linalg.norm(source)
b = np.array(target)/np.linalg.norm(target)
R = create_rotation_matrix(a, b)
result = R @ a

print("Length of a: {})".format(np.sum(np.square(a))))
print("Length of b: {})".format(np.sum(np.square(b))))
print("Length of result: {})".format(np.sum(np.square(result))))

coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size= 0.5)

points1 = Vector3dVector(np.array([[0, 0, 0], a]))
lines1 = Vector2iVector(np.array([[0, 1]]))
line1 = o3d.geometry.LineSet(points1, lines1)
line1.colors = Vector3dVector(np.array([[255, 0, 0]]))

points2 = Vector3dVector(np.array([[0, 0, 0], b]))
lines2 = Vector2iVector(np.array([[0, 1]]))
line2 = o3d.geometry.LineSet(points2, lines2)
line2.colors = Vector3dVector(np.array([[0, 255, 0]]))

points3 = Vector3dVector(np.array([[0, 0, 0], result]))
lines3 = Vector2iVector(np.array([[0, 1]]))
line3 = o3d.geometry.LineSet(points3, lines3)
line3.colors = Vector3dVector(np.array([[0, 0, 255]]))


o3d.visualization.draw_geometries([line1, line2, line3, coord])
