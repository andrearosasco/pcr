from functools import reduce
from pathlib import Path

import torch
import wandb
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
import open3d as o3d
from models.HyperNetwork import HyperNetwork
from configs.local_config import ModelConfig
from torch import nn

from utils.ChamferDistance import chamfer_distance
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


def f1(p, r):
    return 2*((p*r)/(p+r))

if __name__ == "__main__":
    wandb.init(project='eval_pcr')
    # Reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed = TrainConfig.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(0)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    model = HyperNetwork(ModelConfig())

    model.load_state_dict(torch.load("checkpoint/server2.ptc"))
    model.cuda()
    model.eval()

    train_loader = DataLoader(ShapeNet(DataConfig, mode="valid", overfit_mode=TrainConfig.overfit_mode),
                              batch_size=TrainConfig.mb_size,
                              shuffle=False,
                              drop_last=True,
                              num_workers=TrainConfig.num_workers,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=g)

    grid_res = 0.01
    grid = create_3d_grid(-0.5, 0.5, grid_res).to(TrainConfig.device)

    values = range(10)
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('Pastel1'))

    with (Path('../') / 'data' / 'ShapeNetCore.v2' / 'classes.txt').open('r') as f:
        label_names = {l.split()[0]: l.split()[2] for l in f.readlines()}
    precisions = {v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for v in label_names.values()}
    recalls = {v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for v in label_names.values()}
    chamfers = {v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for v in label_names.values()}

    acc_precisions = {v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for v in label_names.values()}
    acc_recalls = {v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for v in label_names.values()}
    acc_chamfers = {v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for v in label_names.values()}

    total = {v: 0 for v in label_names.values()}

    with torch.no_grad():
        for label, partial, mesh in train_loader:

            gt = check_mesh_contains(mesh, grid)
            gt = torch.tensor(gt, device=TrainConfig.device).squeeze(0)

            partial = partial[:1].to(grid.device)  # take just first batch
            fast_weights, _ = model.backbone(partial)
            results = model.sdf(grid, fast_weights)

            thr_pcs = []
            prev_thr = 1

            precision = precisions[label_names[str(label.squeeze().item())]]
            recall = recalls[label_names[str(label.squeeze().item())]]
            chamfer = chamfers[label_names[str(label.squeeze().item())]]
            acc_precision = acc_precisions[label_names[str(label.squeeze().item())]]
            acc_recall = acc_recalls[label_names[str(label.squeeze().item())]]
            acc_chamfer = acc_chamfers[label_names[str(label.squeeze().item())]]
            total[label_names[str(label.squeeze().item())]] += 1


            thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
            for i, threshold in enumerate(thresholds):  # 93 - 77 - 79 - 58

                res = torch.sigmoid(results[0])
                res = torch.logical_and(res > threshold, res < prev_thr)
                pred = grid[0, res.squeeze() == 1.]
                correct = gt[res.squeeze() == 1.]
                true = res[gt.squeeze() == 1.]

                if correct.shape[0] != 0:
                    precision[i] += int(torch.sum(correct).item()) / correct.shape[0]
                    recall[i] += int(torch.sum(true).item()) / torch.sum(gt).item()
                    chamfer[i] += chamfer_distance(grid[:, gt.squeeze().bool(), :].detach().cuda(),
                                      grid[:, res.squeeze().bool(), :].detach().cuda()) * 1000

                res = torch.sigmoid(results[0])
                res = torch.logical_and(res > threshold, res < 1)
                pred = grid[0, res.squeeze() == 1.]
                correct = gt[res.squeeze() == 1.]
                true = res[gt.squeeze() == 1.]

                if correct.shape[0] != 0:
                    acc_precision[i] += int(torch.sum(correct).item()) / correct.shape[0]
                    acc_recall[i] += int(torch.sum(true).item()) / torch.sum(gt).item()
                    acc_chamfer[i] += chamfer_distance(grid[:, gt.squeeze().bool(), :].detach().cuda(),
                                      grid[:, res.squeeze().bool(), :].detach().cuda()) * 1000



                # refine = 0
                # for _ in range(refine):
                #     print('ciao')
                #     side = grid_res / 4
                #     cube = torch.cat([
                #         torch.tensor([[-1, -1, -1]]).repeat(pred.shape[0], 1),
                #         torch.tensor([[1, -1, -1]]).repeat(pred.shape[0], 1),
                #         torch.tensor([[-1, -1, 1]]).repeat(pred.shape[0], 1),
                #         torch.tensor([[1, -1, 1]]).repeat(pred.shape[0], 1),
                #         torch.tensor([[-1, 1, -1]]).repeat(pred.shape[0], 1),
                #         torch.tensor([[1, 1, -1]]).repeat(pred.shape[0], 1),
                #         torch.tensor([[-1, 1, 1]]).repeat(pred.shape[0], 1),
                #         torch.tensor([[1, 1, 1]]).repeat(pred.shape[0], 1),
                #     ]).to(TrainConfig.device)
                #
                #     cube = cube * side
                #     new_pts = pred.repeat(8, 1)
                #
                #     new_pts = new_pts + cube
                #     new_results = model(partial, new_pts.unsqueeze(0))
                #
                #     res = torch.sigmoid(new_results[0] / Temperature)
                #     res = torch.logical_and(res > threshold, res < prev_thr)
                #     pred = new_pts[res.squeeze() == 1.]
                #
                #     print("Found ", pred.shape[0], " points")
                #
                #     pc1 = PointCloud()
                #     pc1.points = Vector3dVector(pred.cpu())
                #     thr_pcs.append(pc1)
                    # o3d.visualization.draw_geometries(thr_pcs)

                # o3d.visualization.draw_geometries([tm] + thr_pcs)

                prev_thr = threshold

            # o3d.visualization.draw_geometries([tm] + thr_pcs,
            #                                           front=[-1, 0.5, 0.5], up=[0, 1, 0], lookat=[0, 0, 0],
            #                                           zoom=1)
        # Recall/Precision on each class and threshold
        for k in precisions.keys():
            print(f'{k} -> Precision {[p / total[k] for p in precisions[k] if p != 0]} - Recall {[r / total[k] for r in recalls[k] if r != 0]}'
                  f'Acc Precision {[p / total[k] for p in acc_precisions[k] if p != 0]} - Acc Recall {[r / total[k] for r in acc_recalls[k] if r != 0]}')

            for i in range(10):
                if precisions[k][i] != 0:
                    wandb.log({f'{k}/Precision':precisions[k][i]/total[k],
                               f'{k}/Acc Precision': acc_precisions[k][i]/total[k],
                               'Threshold': thresholds[i]})
                if recalls[k][i] != 0:
                    wandb.log({f'{k}/Recall': recalls[k][i] / total[k],
                               f'{k}/Acc Recall': acc_recalls[k][i] / total[k],
                               'Threshold': thresholds[i]})

                if precisions[k][i] != 0 and recalls[k][i] != 0:
                    wandb.log({f'{k}/F1': f1(precisions[k][i] / total[k], recalls[k][i] / total[k]),
                               f'{k}/Acc F1': f1(acc_precisions[k][i] / total[k], acc_recalls[k][i] / total[k]),
                               f'{k}/Chamfer': chamfers[k][i]/total[k],
                               f'{k}/Acc Chamfer': acc_chamfers[k][i]/total[k],
                               'Threshold': thresholds[i]})



        # Recall/Precision
        t_prec = [sum([p[i] for p in precisions.values()]) for i in range(10)]
        t_rec = [sum([r[i] for r in recalls.values()]) for i in range(10)]
        t_chamfer = [sum([c[i] for c in chamfers.values()]) for i in range(10)]
        t_tot = [sum(total.values()) for i in range(10)]

        t_acc_prec = [sum([p[i] for p in acc_precisions.values()]) for i in range(10)]
        t_acc_rec = [sum([r[i] for r in acc_recalls.values()]) for i in range(10)]
        t_acc_chamfer = [sum([c[i] for c in acc_chamfers.values()]) for i in range(10)]

        for i in range(10):
            wandb.log({'Total/Precision': t_prec[i] / t_tot[i],
                       'Total/Recall': t_rec[i] / t_tot[i],
                       'Total/Acc Precision': t_acc_prec[i] / t_tot[i],
                       'Total/Acc Recall': t_acc_rec[i] / t_tot[i],
                       'Total/F1': f1(t_prec[i] / t_tot[i], t_rec[i] / t_tot[i]),
                       'Total/Acc F1': f1(t_acc_prec[i] / t_tot[i], t_acc_rec[i] / t_tot[i]),
                       'Total/Chamfer': t_chamfer[i] / t_tot[i],
                       'Total/Acc Chamfer': t_acc_chamfer / t_tot[i],
                       'Threshold': thresholds[i]
                       })

        print(f'Total Precision {[p / t for p, t in zip(t_prec, t_tot)]} - Total Recall {[r / t for r, t in zip(t_rec, t_tot)]}')
        print(f'Total Acc Precision {[p / t for p, t in zip(t_acc_prec, t_tot)]} - Total Acc Recall {[r / t for r, t in zip(t_acc_rec, t_tot)]}')
