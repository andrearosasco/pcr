import open3d as o3d
import random
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import tqdm
import shutil
import os


class DatasetGenerator:
    def __init__(self, n=1000, min_side=1., max_side=5., min_trans=0., max_trans=5., min_rot=0., max_rot=90.):
        self.n = n
        self.min_side = min_side
        self.max_side = max_side
        self.min_trans = min_trans
        self.max_trans = max_trans
        self.min_rot = min_rot
        self.max_rot = max_rot
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=False)
        self.i = 0

    def save_image(self):
        image = self.vis.capture_screen_float_buffer(False)
        image = np.asarray(image) * 255
        image = cv2.resize(image, (264, 264))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(image.shape)
        cv2.imwrite("test.png", image)
        self.vis.register_animation_callback(None)
        self.vis.destroy_window()

    def gen_image(self):
        # Create box
        sizes = []
        for i in range(3):
            sizes.append(random.uniform(self.min_side, self.max_side))
        cube_mesh = o3d.geometry.TriangleMesh.create_box(sizes[0], sizes[1], sizes[2])

        # Translate box
        trans = []
        for i in range(3):
            trans.append(random.uniform(self.min_trans, self.max_trans))
        cube_mesh.translate(trans)

        # Rotate box
        rot = []
        for i in range(3):
            rot.append(random.uniform(self.min_rot, self.max_rot))
        r = R.from_euler('yxz', rot, degrees=True).as_matrix()
        cube_mesh.rotate(r)

        # Color box
        colors = []
        for i in range(3):
            colors.append(random.uniform(0.0, 1.0))
        cube_mesh.paint_uniform_color(colors)

        cube_mesh.compute_vertex_normals()  # TODO ADD TO ADD LIGHT
        self.vis.clear_geometries()
        self.vis.add_geometry(cube_mesh)
        self.vis.capture_screen_image("dataset/" + str(self.i) + "_rgb.png", True)
        self.vis.capture_depth_image("dataset/" + str(self.i) + "_depth.png", True)
        self.vis.capture_depth_point_cloud("dataset/" + str(self.i) + "_partial-pc.pcd", True)

        pcd = cube_mesh.sample_points_uniformly(number_of_points=10000)
        o3d.io.write_point_cloud("dataset/" + str(self.i) + "_complete-pc.pcd", pcd)
        self.i += 1


if __name__ == "__main__":
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")
    os.makedirs("dataset")

    main = DatasetGenerator()
    for t in tqdm.tqdm(range(100)):
        main.gen_image()
