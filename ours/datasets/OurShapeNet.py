import os
import numpy as np
import torch
import torch.utils.data as data
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector
from tqdm import tqdm
from utils.misc import sample_point_cloud
import open3d as o3d


class ShapeNet(data.Dataset):
    def __init__(self, config):
        #  Backbone Input
        self.data_root = config.prep_path
        self.n_points = config.N_POINTS

        # Implicit function input
        self.noise_rate = config.noise_rate
        self.percentage_sampled = config.percentage_sampled
        self.n_files = len([name for name in os.listdir(self.data_root) if os.path.isdir(self.data_root + os.sep + name)])

        print("Found ", self.n_files, " instances")

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
        with open(self.data_root + os.sep + str(idx) + os.sep + "label.txt") as file:
            label = file.read()

        # Extract point cloud from mesh
        tm = o3d.io.read_triangle_mesh(self.data_root + os.sep + str(idx) + os.sep + "model.obj")
        tm = tm.compute_vertex_normals()
        p = tm.sample_points_uniformly(self.n_points)

        # # TODO CONTINUE SEEMS LIKE BY SAMPLING POINTS WE ARE NOT SAMPLING COLORS
        # if os.path.isdir(self.data_root + os.sep + str(idx) + os.sep + "images"):
        #     print("COLORED")  # TODO REMOVE DEBUG
        #     o3d.visualization.draw_geometries([tm])  # TODO REMOVE DEBUG ( VISUALIZE MESH )
        #     o3d.visualization.draw_geometries([p])  # TODO REMOVE DEBUG ( VISUALIZE COMPLETE POINT CLOUD )
        #     print("bau")

        complete_xyz = np.array(p.points)
        complete_xyz = self.pc_norm(complete_xyz)
        complete_xyz = torch.FloatTensor(complete_xyz)
        complete_colors = np.array(p.colors)
        complete_colors = torch.FloatTensor(complete_colors)

        # Use mesh to create input for implicit function
        imp_x, imp_y = sample_point_cloud(tm,
                                          self.noise_rate,
                                          self.percentage_sampled)
        imp_x, imp_y = torch.tensor(imp_x).float(), torch.tensor(imp_y).float()

        return label, (complete_xyz, complete_colors), imp_x, imp_y

    def __len__(self):
        return self.n_files


if __name__ == "__main__":
    from configs.cfg1 import DataConfig
    from torch.utils.data import DataLoader

    dataset = ShapeNet(DataConfig())
    loader = DataLoader(dataset, batch_size=1,
                        shuffle=True,
                        drop_last=True,
                        num_workers=1, pin_memory=True)
    for elem in tqdm(loader):
        lab, complete, x, y = elem

        lab = lab[0]
        print(lab)

        comp_xyz, comp_colors = complete

        x = x.squeeze()
        y = y.squeeze()
        comp_xyz = comp_xyz.squeeze()
        comp_colors = comp_colors.squeeze()

        # TODO REMOVE DEBUG ( VISUALIZE INPUT FOR BACKBONE STILL TO CROP )
        # IF ALL POINTS ARE WHITE, CHANGE TO BLACK
        original = True
        if comp_colors.min() == comp_colors.max() == 1.:
            comp_colors.fill_(0.)
            original = False

        if original:
            pc = PointCloud()
            pc.points = Vector3dVector(comp_xyz)
            pc.colors = Vector3dVector(comp_colors)
            o3d.visualization.draw_geometries([pc], window_name="original" if original else "White converted to black")

        # TODO REMOVE DEBUG ( VISUALIZE IMPLICIT FUNCTION INPUT WITH LABELS )
        # pc = PointCloud()
        # pc.points = Vector3dVector(x)
        # colors = []
        # for v in y:
        #     if v == 0.:
        #         colors.append(np.array([1, 0, 0]))
        #     if v == 1.:
        #         colors.append(np.array([0, 1, 0]))
        # colors = np.stack(colors)
        # pc.colors = Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pc])
