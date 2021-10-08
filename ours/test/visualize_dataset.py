from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from tqdm import tqdm
from datasets.OurShapeNet import ShapeNet
from configs.cfg1 import DataConfig
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d


if __name__ == "__main__":
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
        if len(comp_colors) == 0 or comp_colors.min() == comp_colors.max() == 1.:
            comp_colors.fill_(0.)
            original = False

        pc = PointCloud()
        pc.points = Vector3dVector(comp_xyz)
        pc.colors = Vector3dVector(comp_colors)
        o3d.visualization.draw_geometries([pc], window_name="original" if original else "White converted to black")

        # TODO REMOVE DEBUG ( VISUALIZE IMPLICIT FUNCTION INPUT WITH LABELS )
        pc = PointCloud()
        pc.points = Vector3dVector(x)
        colors = []
        t = 0
        f = 0
        for v in y:
            if v == 0.:
                colors.append(np.array([1, 0, 0]))
                f += 1
            if v == 1.:
                colors.append(np.array([0, 1, 0]))
                t += 1
        colors = np.stack(colors)
        pc.colors = Vector3dVector(colors)
        o3d.open3d.visualization.draw_geometries([pc], window_name=str(t)+" points inside and "+str(f)+" points outside")