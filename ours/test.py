import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
import open3d as o3d
from models.HyperNetwork import HyperNetwork
from configs.local_config import ModelConfig
from torch import nn
from utils.misc import create_3d_grid
from datasets.ShapeNetPOV import ShapeNet
from torch.utils.data import DataLoader
from configs.local_config import DataConfig, TrainConfig
import os
import numpy as np
import random
import time

if __name__ == "__main__":

    # Reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed = TrainConfig.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = HyperNetwork(ModelConfig())
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load("checkpoint/server2.ptc"))
    model.cuda()
    model.eval()



    train_loader = DataLoader(ShapeNet(DataConfig, mode="test", overfit_mode=TrainConfig.overfit_mode),
                              batch_size=TrainConfig.mb_size,
                              shuffle=False,
                              drop_last=True,
                              num_workers=TrainConfig.num_workers,
                              pin_memory=True,
                              generator=g)

    grid_res = 0.01
    grid = create_3d_grid(-0.5, 0.5, grid_res).to(TrainConfig.device)

    with torch.no_grad():
        for label, partial, mesh in train_loader:

            if label.squeeze() == 35:

                partial = partial[:1].to(grid.device)  # take just first batch
                print("Giving to model ", partial.size(1), " points")
                start = time.time()
                results = model(partial, grid)
                end = time.time()
                print(end - start)

                tm = o3d.io.read_triangle_mesh(mesh[0], False)
                o3d.visualization.draw_geometries([tm])

                pc = PointCloud()
                pc.points = Vector3dVector(partial.cpu().squeeze().numpy())
                o3d.visualization.draw_geometries([pc, tm])

                color = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]

                for Temperature in [1]:
                    thr_pcs = []
                    prev_thr = 1
                    for i, threshold in enumerate([0.9, 0.8, 0.7, 0.5, 0.3, 0.1]):

                        res = torch.sigmoid(results[0] / Temperature)
                        res = torch.logical_and(res > threshold, res < prev_thr)
                        pred = grid[0, res.squeeze() == 1.]



                        print("Found ", len(results), " points")

                        pc1 = PointCloud()
                        pc1.points = Vector3dVector(pred.cpu())
                        pc1.paint_uniform_color(color[i])
                        thr_pcs.append(pc1)
                        o3d.visualization.draw_geometries(thr_pcs)

                        for _ in range(2):

                            side = grid_res / 4
                            cube = torch.cat([
                                torch.tensor([[-1, -1, -1]]).repeat(pred.shape[0], 1),
                                torch.tensor([[1, -1, -1]]).repeat(pred.shape[0], 1),
                                torch.tensor([[-1, -1, 1]]).repeat(pred.shape[0], 1),
                                torch.tensor([[1, -1, 1]]).repeat(pred.shape[0], 1),
                                torch.tensor([[-1, 1, -1]]).repeat(pred.shape[0], 1),
                                torch.tensor([[1, 1, -1]]).repeat(pred.shape[0], 1),
                                torch.tensor([[-1, 1, 1]]).repeat(pred.shape[0], 1),
                                torch.tensor([[1, 1, 1]]).repeat(pred.shape[0], 1),
                            ]).to(TrainConfig.device)

                            cube = cube * side
                            new_pts = pred.repeat(8, 1)

                            new_pts = new_pts + cube
                            new_results = model(partial, new_pts.unsqueeze(0))

                            res = torch.sigmoid(new_results[0] / Temperature)
                            res = torch.logical_and(res > threshold, res < prev_thr)
                            pred = new_pts[res.squeeze() == 1.]

                            print("Found ", pred.shape[0], " points")

                            pc1 = PointCloud()
                            pc1.points = Vector3dVector(pred.cpu())
                            thr_pcs.append(pc1)
                            o3d.visualization.draw_geometries(thr_pcs)

                        o3d.visualization.draw_geometries([tm] + thr_pcs)

                        prev_thr = threshold