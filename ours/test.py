import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
import open3d as o3d
from models.HyperNetwork import HyperNetwork
from configs import ModelConfig
from torch import nn
from utils.misc import create_3d_grid
from datasets.ShapeNetPOV import ShapeNet
from torch.utils.data import DataLoader
from configs import DataConfig, TrainConfig
import os
import numpy as np
import random
import time

if __name__ == "__main__":

    model = HyperNetwork(ModelConfig())
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load("checkpoint/server2.ptc"))
    model.cuda()
    model.eval()

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

    train_loader = DataLoader(ShapeNet(DataConfig, mode="train", overfit_mode=TrainConfig.overfit_mode),
                              batch_size=TrainConfig.mb_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=TrainConfig.num_workers,
                              pin_memory=True,
                              generator=g)

    grid = create_3d_grid(-0.5, 0.5, 0.01).to(TrainConfig.device)

    with torch.no_grad():
        for label, partial, data, imp_x, imp_y, padding_length in train_loader:
            partial = partial[:1].to(grid.device)  # take just first batch
            print("Giving to model ", partial.size(1), " points")
            start = time.time()
            results = model(partial, grid)
            end = time.time()
            print(end - start)
            break

    pc = PointCloud()
    pc.points = Vector3dVector(data[0])
    o3d.visualization.draw_geometries([pc])

    results = results[0]
    grid = grid[0]
    results = results > 0.5
    results = torch.cat((grid, results), dim=-1)
    results = results[results[..., -1] == 1.]
    results = results[:, :3]

    print("Found ", len(results), " points")

    pc = PointCloud()
    pc.points = Vector3dVector(results.cpu())
    o3d.visualization.draw_geometries([pc])
