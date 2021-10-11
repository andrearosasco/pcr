import os
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.utils.data as data
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries

from utils.misc import sample_point_cloud


class ShapeNet(data.Dataset):
    def __init__(self, config):
        #  Backbone Input
        self.data_root = Path(config.dataset_path)
        self.mode = config.mode
        self.n_points = config.N_POINTS

        # Implicit function input
        self.noise_rate = config.noise_rate
        self.percentage_sampled = config.percentage_sampled

        with (self.data_root / f'{self.mode}.txt').open('r') as file:
            lines = file.readlines()

        self.samples = lines
        self.n_samples = len(lines)

        with (self.data_root / 'classes.txt').open('r') as file:
            self.labels_map = {l.split()[1]: l.split()[0] for l in file.readlines()}

        print("Found ", self.n_samples, " instances")

    @staticmethod
    def pc_norm(pcs):
        """ pc: NxC, return NxC """
        centroid = np.mean(pcs, axis=0)
        pcs = pcs - centroid
        m = np.max(np.sqrt(np.sum(pcs ** 2, axis=1)))
        pcs = pcs / m
        return pcs

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y
        # Get label
        dir_path = self.data_root / self.samples[idx].strip()
        label = self.labels_map[dir_path.parent.name]

        # Extract point cloud from mesh
        tm = o3d.io.read_triangle_mesh(str(dir_path / 'models/model_normalized.obj'), True)

        complete_pcd = tm.sample_points_uniformly(self.n_points)

        diameter = np.linalg.norm(
            np.asarray(complete_pcd.get_max_bound()) - np.asarray(complete_pcd.get_min_bound()))
        camera = [1, 0, diameter]
        radius = diameter * 100
        _, pt_map = complete_pcd.hidden_point_removal(camera, radius * 4)
        partial_pcd = complete_pcd.select_by_index(pt_map)
        # TODO Start Remove
        img_width = 640
        img_height = 480
        render = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
        # Optionally set the camera field of view (to zoom in a bit)
        vertical_field_of_view = 15.0  # between 5 and 90 degrees
        aspect_ratio = img_width / img_height  # azimuth over elevation
        near_plane = 0.1
        far_plane = 50.0
        fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
        render.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

        # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
        center = [0, 0, 0]  # look_at target
        eye = [1, 0, diameter]  # camera position
        up = [0, 1, 0]  # camera orientation
        render.scene.camera.look_at(center, eye, up)

        # Read the image into a variable
        img_o3d = render.render_to_image()

        # Display the image in a separate window
        # (Note: OpenCV expects the color in BGR format, so swop red and blue.)
        img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
        cv2.imshow("Preview window", img_cv2)
        cv2.waitKey()  # np.ndarray(np.float64([[1, 0, diameter]]).T, dtype=np.int)
        # TODO End Remove
        partial_pcd = np.array(partial_pcd.points)
        # partial_pcd = self.pc_norm(partial_pcd)
        partial_pcd = torch.FloatTensor(partial_pcd)
        complete_pcd = np.array(complete_pcd.points)
        # complete_pcd = self.pc_norm(complete_pcd)
        complete_pcd = torch.FloatTensor(complete_pcd)

        imp_x, imp_y = sample_point_cloud(tm,
                                          self.noise_rate,
                                          self.percentage_sampled)
        imp_x, imp_y = torch.tensor(imp_x).float(), torch.tensor(imp_y).bool().float().bool().float()  # TODO oh god..

        # return label, (complete_xyz, complete_colors), imp_x, imp_y
        return label, complete_pcd, partial_pcd, imp_x, imp_y

    def __len__(self):
        return self.n_samples



if __name__ == "__main__":
    from configs.cfg1 import DataConfig
    iterator = ShapeNet(DataConfig)
    for elem in iterator:
        lab, comp, part, x, y = elem
        print(lab)
        pc = PointCloud()
        pc.points = Vector3dVector(part)
        o3d.visualization.draw_geometries([pc])

        pc = PointCloud()
        pc.points = Vector3dVector(comp)
        colors = []
        t = 0
        f = 0
        for point in y:
            if point == 1:
                colors.append(np.array([0, 1, 0]))
                t += 1
            else:
                colors.append(np.array([1, 0, 0]))
                f += 1
        # colors = np.stack(colors)
        # colors = Vector3dVector(colors)
        # pc.colors = colors
        o3d.visualization.draw_geometries([pc], window_name="Green "+str(t) + ", red: " + str(f))
