from pathlib import Path

import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
import open3d as o3d
from models.HyperNetwork import HyperNetwork
from configs.local_config import ModelConfig
from torch import nn
from utils.misc import create_3d_grid, check_mesh_contains
from datasets.ShapeNetPOV import ShapeNet
from torch.utils.data import DataLoader
from configs.local_config import DataConfig, TrainConfig
import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


class Grid:

    def __init__(self, min, max, res, mb_size):
        self.min = min
        self.max = max
        self.res = res
        self.mb_size = mb_size

        self.no_points = ((max - min) / res)**3
        self.step = (self.mb_size / self.no_points) * (max - min)

        self.i = 0

    def __next__(self):
        bs = 1
        if self.min+self.step*(self.i+1) > self.max:
            raise StopIteration

        x_range = torch.FloatTensor(np.arange(self.min+self.step*self.i, self.min+self.step*(self.i+1), self.res))
        y_range = torch.FloatTensor(np.arange(self.min, self.max, self.res))
        z_range = torch.FloatTensor(np.arange(self.min, self.max, self.res))
        grid_2d = torch.cartesian_prod(x_range, y_range)
        grid_2d = grid_2d.repeat(y_range.shape[0], 1)
        z_repeated = z_range.unsqueeze(1).T.repeat(y_range.shape[0] * x_range.shape[0], 1).T.reshape(-1)[..., None]
        grid_3d = torch.cat((grid_2d, z_repeated), dim=-1)
        grid_3d = grid_3d.unsqueeze(0).repeat(bs, 1, 1)

        self.i+=1
        return grid_3d

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.no_points) // self.mb_size


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

    grid = Grid(-0.4, 0.4, 0.0001, 10000)

    values = range(10)
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('Pastel1'))

    with (Path('../') / 'data' / 'ShapeNetCore.v2' / 'classes.txt').open('r') as f:
        label_names = {l.split()[0]: l.split()[2] for l in f.readlines()}
    precisions = {v: [0, 0, 0, 0, 0] for v in label_names.values()}
    total = {v: 0 for v in label_names.values()}

    with torch.no_grad():
        for label, partial, mesh in train_loader:

            if label.squeeze() == 35:
                out = []
                gts = []
                partial = partial[:1].to(TrainConfig.device)  # take just first batch
                print("Giving to model ", partial.size(1), " points")
                start = time.time()
                fast_weights, _ = model.backbone(partial)
                print('Backbone', time.time() - start)
                for x in grid:
                    x = x.to(TrainConfig.device)
                    gt = check_mesh_contains(mesh, x)
                    gts.append(gt.squeeze(0))

                    start = time.time()
                    results = model.sdf(x, fast_weights)
                    print('Implicit Function', time.time() - start)
                    end = time.time()

                    res = torch.sigmoid(results.cpu()[0])
                    res = torch.logical_and(res > 0.9, res <1)

                    out.append(x[0, res.squeeze() == 1.])

                thr_pcs = []
                prev_thr = 1

                precision = precisions[label_names[str(label.squeeze().item())]]
                total[label_names[str(label.squeeze().item())]] += 1

            pcs = []
            for i in range(len(out)):
                pc1 = PointCloud()
                pc1.points = Vector3dVector(out[i].cpu())
                # colorVal = scalarMap.to_rgba(values[i])
                # pc1.paint_uniform_color(colorVal[:3])
                pcs.append(pc1)
            o3d.visualization.draw_geometries(pcs,
                                              front=[-1, 0.5, 0.5], up=[0, 1, 0], lookat=[0, 0, 0],
                                              zoom=1)


            # o3d.visualization.draw_geometries([tm] + thr_pcs,
            #                                           front=[-1, 0.5, 0.5], up=[0, 1, 0], lookat=[0, 0, 0],
            #                                           zoom=1)
        for k in precisions.keys():
            print(f'{k} -> {[p / total[k] for p in precisions[k] if p != 0]}')