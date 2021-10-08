import open3d as o3d
import cv2


# tm = o3d.io.read_triangle_mesh("capsule.obj", True)
# o3d.visualization.draw_geometries([tm])

tm = o3d.io.read_triangle_mesh("../datasets/labelled_rgb_meshes/1/model.obj", True)
o3d.visualization.draw_geometries([tm])
