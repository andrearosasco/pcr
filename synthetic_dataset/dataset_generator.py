import open3d as o3d
import random
from scipy.spatial.transform import Rotation as R
import tqdm
import shutil
import os


class DatasetGenerator:
    def __init__(self, n=1000, min_side=1., max_side=5., min_trans=0., max_trans=5., min_rot=0., max_rot=90.,
                 n_points_partial=1000000, min_points=1000, ycb=True):
        self.n = n
        self.min_side = min_side
        self.max_side = max_side
        self.min_trans = min_trans
        self.max_trans = max_trans
        self.min_rot = min_rot
        self.max_rot = max_rot
        self.n_points_partial = n_points_partial
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=False)
        self.min_points = 1000
        self.i = 0
        self.ycb = ycb
        self.min_points = min_points
        if self.ycb:
            with open('names.txt') as f:
                self.objects = f.read().splitlines()

    # def save_image(self):
    #     image = self.vis.capture_screen_float_buffer(False)
    #     image = np.asarray(image) * 255
    #     image = cv2.resize(image, (264, 264))
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     print(image.shape)
    #     cv2.imwrite("test.png", image)
    #     self.vis.register_animation_callback(None)
    #     self.vis.destroy_window()

    def gen_image(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=False)

        # Load ycb object
        if self.ycb:
            idx = random.randint(0, len(self.objects))
            filename = "ycb/" + self.objects[idx] + "/clouds/merged_cloud.ply"
            try:
                cube_mesh = o3d.io.read_point_cloud(filename, format='ply')
            except Exception as e:
                print(str(e))
                return False
            if len(cube_mesh.points) < self.min_points:
                print(filename + " has only " + str(len(cube_mesh.points)) + " points")
                return False
        # Or create box
        else:
            sizes = []
            for i in range(3):
                sizes.append(random.uniform(self.min_side, self.max_side))
            cube_mesh = o3d.geometry.TriangleMesh.create_box(sizes[0], sizes[1], sizes[2])

        # Translate
        trans = []
        for i in range(3):
            trans.append(random.uniform(self.min_trans, self.max_trans))
        cube_mesh.translate(trans)

        # Rotate
        rot = []
        for i in range(3):
            rot.append(random.uniform(self.min_rot, self.max_rot))
        r = R.from_euler('yxz', rot, degrees=True).as_matrix()
        cube_mesh.rotate(r)

        # Color (if box)
        if not self.ycb:
            colors = []
            for i in range(3):
                colors.append(random.uniform(0.0, 1.0))
            cube_mesh.paint_uniform_color(colors)
            cube_mesh.compute_vertex_normals()

        self.vis.clear_geometries()
        self.vis.add_geometry(cube_mesh)
        self.vis.capture_screen_image("dataset/" + str(self.i) + "_rgb.png", True)
        self.vis.capture_depth_image("dataset/" + str(self.i) + "_depth.png", True)
        self.vis.capture_depth_point_cloud("dataset/" + str(self.i) + "_partial-pc.pcd", True)

        # Sample points if not box
        if not self.ycb:
            cube_mesh = cube_mesh.sample_points_uniformly(number_of_points=self.n_points_partial)

        # Save complete (sampled) point cloud
        try:
            o3d.io.write_point_cloud("dataset/" + str(self.i) + "_complete-pc.pcd", cube_mesh)
        except Exception as e:
            print(str(e))
            return False

        self.i += 1
        return True


if __name__ == "__main__":
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")
    os.makedirs("dataset")

    progress_bar = tqdm.tqdm(total=10)
    main = DatasetGenerator()
    while main.i < 10:
        if main.gen_image():
            progress_bar.update(1)
