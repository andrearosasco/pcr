from configs.cfg1 import DataConfig
from datasets.ShapeNet55Dataset import ShapeNet
from torch.utils.data import DataLoader
import open3d as o3d
import torch


if __name__ == "__main__":
    config = DataConfig()
    config.DATA_PATH = str("..\\") + str(config.DATA_PATH)
    config.PC_PATH = str("..\\") + str(config.PC_PATH)
    dataset = ShapeNet(config)
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=False,
                            drop_last=True,
                            num_workers=10, pin_memory=True)
    for idx, (taxonomy_ids, model_ids, data, imp_x, imp_y) in enumerate(dataloader):

        # GET TRUE AND FALSE POINTS
        true = []
        for point, value in zip(imp_x[0], imp_y[0].unsqueeze(-1)):
            if value.squeeze().item() == 1.:
                true.append(point)
        if len(true) > 0:
            true = torch.stack(true)
        else:
            true = torch.zeros(1, 3)

        false = []
        for point, value in zip(imp_x[0], imp_y[0].unsqueeze(-1)):
            if value.squeeze().item() == 0.:
                false.append(point)
        if len(true) > 0:
            false = torch.stack(false)
        else:
            false = torch.zeros(1, 3)

        # WHOLE
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(imp_x[0])
        # o3d.visualization.draw_geometries([pcd])

        # TRUE AFTER NOISE AND VOXEL
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(true)
        # o3d.visualization.draw_geometries([pcd])

        # FALSE AFTER NOISE AND VOXEL
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(false)
        # o3d.visualization.draw_geometries([pcd])

        # WITH CLASS
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(imp_x[0])
        colors = torch.zeros_like(imp_x[0])
        colors[..., 0] = imp_y[0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
