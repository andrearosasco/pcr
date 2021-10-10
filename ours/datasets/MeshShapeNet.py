import os
import numpy as np
import torch
import torch.utils.data as data
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector

from utils.misc import sample_point_cloud


class ShapeNet(data.Dataset):
    def __init__(self, config, mode):
        #  Backbone Input
        self.data_root = config.prep_path
        self.n_points = config.N_POINTS
        self.mode = mode

        # Implicit function input
        self.noise_rate = config.noise_rate
        self.percentage_sampled = config.percentage_sampled

        self.txts_path = config.txts_path
        self.train_txt_path = self.txts_path + os.sep + self.mode + ".txt"

        with open(self.train_txt_path, "r") as file:
            lines = file.readlines()

        self.train_paths = lines
        self.n_samples = len(lines)

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
        # RETURN MESH,
        # Get label
        dir_path = self.data_root + os.sep + self.train_paths[idx].strip()
        with open(dir_path + os.sep + "label.txt") as file:
            label = file.read()

        # Extract point cloud from mesh
        tm = o3d.io.read_triangle_mesh(dir_path + os.sep + "model.obj", True)
        # # TODO TRY
        # import cv2
        # img1 = cv2.imread(self.data_root + os.sep + str(idx) + os.sep + "images/texture0.jpg")
        # img2 = cv2.imread(self.data_root + os.sep + str(idx) + os.sep + "images/texture1.jpg")
        # tm.textures = [o3d.geometry.Image(img1), o3d.geometry.Image(img2)]
        # TODO END TRY
        # tm.compute_vertex_normals()
        p = tm.sample_points_uniformly(self.n_points)

        # TODO CONTINUE SEEMS LIKE BY SAMPLING POINTS WE ARE NOT SAMPLING COLORS
        # if os.path.isdir(dir_path + os.sep + "images"):
        #     print("COLORED")  # TODO REMOVE DEBUG
        #     print(tm.has_textures())
        #     o3d.visualization.draw_geometries([tm])  # TODO REMOVE DEBUG
        #     # o3d.visualization.draw_geometries([tm])  # TODO REMOVE DEBUG ( VISUALIZE MESH )
        #     o3d.visualization.draw_geometries([p])  # TODO REMOVE DEBUG ( VISUALIZE COMPLETE POINT CLOUD )
        #     print("bau")

        complete_xyz = np.array(p.points)
        complete_xyz = self.pc_norm(complete_xyz)
        complete_xyz = torch.FloatTensor(complete_xyz)
        # complete_colors = np.array(p.colors)
        # complete_colors = torch.FloatTensor(complete_colors)

        # Use mesh to create input for implicit function
        imp_x, imp_y = sample_point_cloud(tm,
                                          self.noise_rate,
                                          self.percentage_sampled)
        imp_x, imp_y = torch.tensor(imp_x).float(), torch.tensor(imp_y).bool().float().bool().float()  # TODO oh god..

        # return label, (complete_xyz, complete_colors), imp_x, imp_y
        return label, complete_xyz, imp_x, imp_y

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    from configs.cfg1 import DataConfig
    iterator = ShapeNet(DataConfig)
    for elem in iterator:
        lab, comp, x, y = elem
        print(lab)
        pc = PointCloud()
        pc.points = Vector3dVector(comp)
        o3d.visualization.draw_geometries([pc])

        pc = PointCloud()
        pc.points = Vector3dVector(x)
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
        colors = np.stack(colors)
        colors = Vector3dVector(colors)
        pc.colors = colors
        o3d.visualization.draw_geometries([pc], window_name="Green "+str(t) + ", red: " + str(f))
