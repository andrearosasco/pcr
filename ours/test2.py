import torch
from open3d.cpu.pybind.geometry import PointCloud, LineSet
from open3d.cpu.pybind.utility import Vector3dVector, Vector2iVector
import open3d as o3d
from open3d.cpu.pybind.visualization import draw_geometries
from torch.nn import BCEWithLogitsLoss

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
    grid = create_3d_grid(-0.5, 0.5, grid_res)

    loss_function = BCEWithLogitsLoss(reduction='mean')

    for label, partial, mesh in train_loader:

        if label.squeeze() == 35:
            gt = check_mesh_contains(mesh, grid)
            gt = torch.tensor(gt, device=TrainConfig.device)

            partial = partial[:1].to(TrainConfig.device)

            fast_weights, _ = model.backbone(partial)
            logits = model.sdf(grid.to(TrainConfig.device), fast_weights)

            idx = (torch.sigmoid(logits.squeeze(-1).detach()) > 0.5).squeeze().bool()
            gt = gt[:, idx, :]

            refined_pred = torch.tensor(grid[:, idx, :].cpu().detach().numpy(), device=TrainConfig.device,
                                        requires_grad=True)

            prev_num = [-1] * 10

            for step in range(1000):

                partial = partial[:1].to(TrainConfig.device)  # take just first batch
                fast_weights, _ = model.backbone(partial)
                results = model.sdf(refined_pred, fast_weights)

                  # Without axis it does not work with just one batch

                loss_value = loss_function(results[..., 0], torch.ones_like(gt[..., 0], dtype=torch.float32))
                model.zero_grad()
                loss_value.backward(inputs=[refined_pred])

                print('Loss ', loss_value.item())

                grad = refined_pred.grad.data
                refined_pred = refined_pred - (1 * grad)

                with torch.no_grad():
                    prev_thr = 1

                    for i, t in enumerate([0.9, 0.8, 0.7, 0.6, .5, .4, .3, .2, .1, 0]):
                        # Have to be recomputed because points were moved
                        truth = check_mesh_contains(mesh, refined_pred)
                        truth = torch.tensor(truth, device=TrainConfig.device)

                        res = torch.sigmoid(results.squeeze().detach())
                        eval_idx = torch.logical_and(res > t, res < prev_thr).bool()
                        pred = refined_pred[0, eval_idx]
                        correct = truth[0, eval_idx]

                        print(prev_thr, '> p >', t, end=' ')
                        if prev_num[i] >= 0:
                            print('- Added ', correct.shape[0] - prev_num[i], end=' ')
                        else:
                            print('- Added 0', end=' ')
                        prev_num[i] = correct.shape[0]
                        if correct.shape[0] != 0:
                            print(f'- Precision {int(torch.sum(correct).item()) / correct.shape[0]} Total {correct.shape[0]}')
                        else:
                            print('- Precision 0.0')

                        prev_thr = t

                refined_pred = torch.tensor(refined_pred.cpu().detach().numpy(), device=TrainConfig.device,
                                            requires_grad=True)
                print()


            aux1 = PointCloud()
            idx = (torch.sigmoid(logits.squeeze(-1).detach()) > 0.5).squeeze().bool()
            aux1.points = Vector3dVector(grid[0, idx, :].cpu().detach().numpy())
            draw_geometries([aux1])

            res = torch.sigmoid(results.squeeze().detach())
            eval_idx = torch.logical_and(res > 0, res < 1).bool()
            pred = refined_pred[0, eval_idx]
            aux2 = PointCloud()
            aux2.points = Vector3dVector(pred.detach().cpu().numpy())
            draw_geometries([aux2])

            v = torch.vstack([grid[0, idx, :], pred.cpu()]).detach().numpy()
            e = np.arange(0, v.shape[0]).reshape([2, v.shape[0] // 2]).T
            lines = LineSet()
            lines.points = Vector3dVector(v)
            lines.lines = Vector2iVector(e)

            aux1.paint_uniform_color([1, 0, 0])
            aux2.paint_uniform_color([0, 1, 0])
            draw_geometries([aux1, aux2, lines])

            tm = o3d.io.read_triangle_mesh(mesh[0], False)
            draw_geometries([tm, aux2])


            exit()
                # print('Backprop and update', time.time() - start)
                #
                # tm = o3d.io.read_triangle_mesh(mesh[0], False)
                # o3d.visualization.draw_geometries([tm])
                #
                # pc = PointCloud()
                # pc.points = Vector3dVector(partial.cpu().squeeze().numpy())
                # o3d.visualization.draw_geometries([pc, tm])
                #
                # thr_pcs = []
                # prev_thr = 1
                # threshold = 0.5
                #
                # res = torch.sigmoid(results[0])
                # res = torch.logical_and(res > threshold, res < prev_thr)
                # pred = grid[0, res.squeeze() == 1.]
                # gt = gt.squeeze()
                # correct = gt[res.squeeze() == 1.]
                #
                # print(f'{int(sum(correct).item())} / {correct.shape[0]} = {int(sum(correct).item()) / correct.shape[0]}')
                #
                # pc1 = PointCloud()
                # pc1.points = Vector3dVector(pred.detach().cpu().numpy())
                # thr_pcs.append(pc1)
                # o3d.visualization.draw_geometries(thr_pcs)
                #
                #
                # o3d.visualization.draw_geometries([tm] + thr_pcs)
                #
                # prev_thr = threshold