from abc import ABC
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from open3d.cpu.pybind.visualization import draw_geometries

from datasets.PCNDataset import PCN
from utils.chamfer import ChamferDistanceL2, ChamferDistanceL1
from utils.metrics import chamfer_batch, chamfer_batch_pc
from utils.reproducibility import get_init_fn, get_generator
from utils.sgdiff import DifferentiableSGD
from .Backbone import BackBone
from .Decoder2 import Decoder as Decoder2
from .Decoder import Decoder
from .ImplicitFunction import ImplicitFunction

try:
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cpu.pybind.geometry import PointCloud
    # from open3d.open3d.utility import Vector3dVector, Vector3iVector
    # from open3d.open3d.geometry import PointCloud

from configs import Config

from torchmetrics import Accuracy, Precision, Recall, F1, MeanMetric
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.ShapeNetPOVDepth import ShapeNet as Dataset

from utils.misc import create_3d_grid, check_mesh_contains, sample_point_cloud, read_mesh_debug, read_mesh, \
    sample_point_cloud_pc, check_occupancy
import open3d as o3d
import pytorch_lightning as pl
import wandb


class PCRNetwork(pl.LightningModule, ABC):

    def __init__(self, config):
        super().__init__()
        self.backbone = BackBone(config)
        self.sdf = ImplicitFunction(config)
        self.decoder = Decoder(self.sdf)
        self.train_decoder = Decoder2(self.sdf)

        # self.apply(self._init_weights)
        for parameter in self.backbone.transformer.parameters():
            if len(parameter.size()) > 2:
                torch.nn.init.xavier_uniform_(parameter)

        self.accuracy = Accuracy()
        self.precision_ = Precision()
        self.recall = Recall()
        self.f1 = F1()
        self.avg_loss = MeanMetric()
        self.avg_chamfer = MeanMetric()

        self.pre_adpt_chamfer = MeanMetric()
        self.train_cd_l1 = MeanMetric()

        self.training_set, self.valid_set = None, None

        self.rt_setup = False

        self.cls_count = defaultdict(lambda: 0)
        self.cls_cd = defaultdict(lambda: 0)

        self.cd = []
        self.labels = []

    # def _init_weights(self, m):
    #     if isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm1d):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight.data)
    #
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def prepare_data(self):
        # self.training_set = Dataset(Config, mode=f"{Config.Data.mode}/train")
        # self.valid_set = Dataset(Config, mode=f"{Config.Data.mode}/valid")

        self.training_set = PCN(subset="train")
        self.valid_set = PCN(subset="val")
        print(len(self.valid_set))

    def train_dataloader(self):
        dl = DataLoader(self.training_set,
                        batch_size=Config.Train.mb_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=Config.General.num_workers,
                        pin_memory=True,
                        # worker_init_fn=get_init_fn(TrainConfig.seed),
                        # generator=get_generator(TrainConfig.seed)
                        )

        return dl

    def val_dataloader(self):
        dl = DataLoader(
            self.valid_set,
            shuffle=False,
            batch_size=Config.Eval.mb_size,
            drop_last=False,
            num_workers=Config.General.num_workers,
            pin_memory=True)
        print(len(dl))
        print(Config.Eval.mb_size)
        return dl

    def forward(self, partial, object_id=None, step=0.01):
        # if not self.rt_setup:
        #     self.rt_setup = True
        #     x = torch.ones((1, 2024, 3)).cuda()
        # #     # self.backbone_tr = torch2trt(self.backbone, [x], use_onnx=True)
        #     torch.onnx.export(self.backbone, x, 'pcr.onnx', input_names=['input'], output_names=['output'],
        #                      )
            # y = create_3d_grid(batch_size=partial.shape[0], step=step).to(TrainConfig.device)
            # self.sdf_tr = torch2trt(self.sdf, [y])

        samples = create_3d_grid(batch_size=partial.shape[0], step=step).to(Config.General.device)

        fast_weights, _ = self.backbone(partial)
        prediction = torch.sigmoid(self.sdf(samples, fast_weights))

        return samples[prediction.squeeze(-1) > 0.5], fast_weights

    def configure_optimizers(self):
        optimizer = Config.Train.optimizer(self.parameters(), lr=Config.Train.lr, weight_decay=Config.Train.wd)
        return optimizer

    def on_train_epoch_start(self):
        self.accuracy.reset(), self.precision_.reset(), self.recall.reset()
        self.f1.reset(), self.avg_loss.reset()

    def training_step(self, batch, batch_idx):
        _, _, (partial, ground_truth) = batch

        fast_weights, _ = self.backbone(partial)
        ### Adaptation
        if Config.Train.adaptation:
            adaptation_steps = 10

            fast_weights = [[t.requires_grad_(True) for t in l] for l in fast_weights]

            for _ in range(adaptation_steps):
                optim = DifferentiableSGD(sum(fast_weights, []), lr=0.1, momentum=0.9)  # the sum flatten the list of list

                out = self.sdf(partial, fast_weights)
                # The loss function also computes the sigmoid
                loss = F.binary_cross_entropy_with_logits(out, torch.ones(partial.shape[:2] + (1,), device=Config.General.device))

                loss.backward(inputs=sum(fast_weights, []), retain_graph=True)
                fast_weights = optim.step()
                fast_weights = [[fast_weights[i], fast_weights[i + 1], fast_weights[i + 2]] for i in range(0, 12, 3)]

        if Config.Train.chamfer:
            pred = self.train_decoder(fast_weights)
            loss = ChamferDistanceL2()(pred, ground_truth)

            return {'loss': loss, 'pred': pred, 'target': ground_truth}

        else:
            # mod_samples = self.decoder(fast_weights)

            samples, target = sample_point_cloud_pc(ground_truth, n_points=Config.Data.implicit_input_dimension,
                                                       dist=Config.Data.dist,
                                                       noise_rate=Config.Data.noise_rate,
                                                       tolerance=Config.Data.tolerance)
            target = target.float()
            # mod_target = check_occupancy(ground_truth, mod_samples, voxel_size=Config.Data.tolerance)
            # mod_target = mod_target.float()

            # Rebalancing
            # p = (torch.sum(mod_target, dim=1) / mod_target.shape[1]).mean()
            # n = 1 - p
            #
            # reb_samples, reb_target = sample_point_cloud_pc(ground_truth, n_points=Config.Data.implicit_input_dimension,
            #                                         dist=[p*0.2, p*0.8, n],
            #                                         noise_rate=Config.Data.noise_rate,
            #                                         tolerance=Config.Data.tolerance)

            # samples = torch.cat([mod_samples, reb_samples], dim=1)
            # target = torch.cat([mod_target, reb_target], dim=1)

            out = self.sdf(samples, fast_weights)
            pred = torch.sigmoid(out.detach()).cpu()

            loss = F.binary_cross_entropy_with_logits(out, target.unsqueeze(-1))
            target = target.unsqueeze(-1).int()

            # if torch.sum(torch.prod(ground_truth == 0.0, dim=2) != 0, dim=1).item():
            #     print()
            #     pass

            # for p, l, g in zip(samples, target, ground_truth):
            #     aux1 = PointCloud(points=Vector3dVector(p[l.bool().squeeze()].cpu().numpy()))
            #     aux1.paint_uniform_color([0, 1, 0])
            #     aux2 = PointCloud(points=Vector3dVector(p[~l.bool().squeeze()].cpu().numpy()))
            #     aux2.paint_uniform_color([1, 0, 0])
            #     aux3 = PointCloud(points=Vector3dVector(g.cpu().numpy()))
            #     aux3.paint_uniform_color([0, 0, 1])
            #
            #     o3d.pybind.visualization.draw_geometries([aux1, aux3])
            #     o3d.pybind.visualization.draw_geometries([aux2, aux3])
            #     o3d.pybind.visualization.draw_geometries([aux1, aux2])
            #     break

            return {'loss': loss, 'pred': pred.detach().cpu(), 'target': target.detach().cpu()}

    @torch.no_grad()
    def training_step_end(self, output):
        pred, trgt = output['pred'], output['target']

        # This log the metrics on the current batch and accumulate it in the average
        if Config.Train.chamfer:
            cd = ChamferDistanceL1()(pred, trgt)
            self.log('train/chamfer', self.train_cd_l1(cd))
            output['pred'], output['target'] = pred.cpu(), trgt.cpu()
        else:
            self.log('train/accuracy', self.accuracy(pred, trgt))
            self.log('train/precision', self.precision_(pred, trgt))
            self.log('train/recall', self.recall(pred, trgt))
            self.log('train/f1', self.f1(pred, trgt))

        self.log('train/loss', self.avg_loss(output['loss'].detach().cpu()))

    def on_validation_epoch_start(self):
        self.accuracy.reset(), self.precision_.reset(), self.recall.reset()
        self.f1.reset(), self.avg_loss.reset(), self.avg_chamfer.reset(), self.pre_adpt_chamfer.reset()

    def validation_step(self, batch, batch_idx):
        class_id, label, (partial, ground_truth) = batch

        # The sampling with "sample_point_cloud" simulate the sampling used during training
        # This is useful as we always get ground truths sampling on the meshes but it doesn't reflect
        #   how the algorithm will work after deployment

        samples2, occupancy2 = sample_point_cloud_pc(ground_truth, n_points=Config.Data.implicit_input_dimension, dist=Config.Data.dist,
                                  noise_rate=Config.Data.noise_rate, tolerance=Config.Data.tolerance)

        ############# INFERENCE #############
        fast_weights, _ = self.backbone(partial)
        pc1_pre = self.decoder(fast_weights)

        ### Adaptation
        if Config.Train.adaptation:
            with torch.enable_grad():
                adaptation_steps = 10

                fast_weights = [[t.requires_grad_(True) for t in l] for l in fast_weights]

                for _ in range(adaptation_steps):
                    optim = DifferentiableSGD(sum(fast_weights, []), lr=0.1,
                                              momentum=0.9)  # the sum flatten the list of list

                    out = self.sdf(partial, fast_weights)
                    loss = F.binary_cross_entropy_with_logits(out, torch.ones(partial.shape[:2] + (1,),
                                                                              device=Config.General.device))

                    loss.backward(inputs=sum(fast_weights, []), retain_graph=True)
                    fast_weights = optim.step()
                    fast_weights = [[fast_weights[i], fast_weights[i + 1], fast_weights[i + 2]] for i in range(0, 12, 3)]

        pc1 = self.decoder(fast_weights)
        out2 = torch.sigmoid(self.sdf(samples2, fast_weights))

        # target = check_occupancy(ground_truth, pc1, voxel_size=Config.Data.tolerance)

        # if label[0] == '545672cd928e85e7d706ecb3379aa341':
        #     print()
        #     pass
        # for p, l, g in zip(pc1, target, ground_truth):
        #     self.cls_count[class_id[0]] += 1
        #     cd = chamfer_batch_pc(p.unsqueeze(0), g.unsqueeze(0)).item()
        #     self.cls_cd[class_id[0]] += torch.sum(torch.prod(ground_truth == 0.0, dim=2) != 0, dim=1).item()  # cd
            # if torch.sum(torch.sum(ground_truth == 0.0, dim=2) != 0, dim=1).item() != 0:
            #     print()
            #     pass

            # self.cd.append(cd)
            # self.labels.append(label)
            # aux1 = PointCloud(points=Vector3dVector(p[l.bool().squeeze()].cpu().numpy()))
            # aux1.paint_uniform_color([0, 1, 0])
            # aux2 = PointCloud(points=Vector3dVector(p[~l.bool().squeeze()].cpu().numpy()))
            # aux2.paint_uniform_color([1, 0, 0])
            # aux3 = PointCloud(points=Vector3dVector(g.cpu().numpy()))
            # aux3.paint_uniform_color([0, 0, 1])
            #
            # o3d.pybind.visualization.draw_geometries([aux1, aux2, aux3])

        return {'pc1': pc1, 'pre_pc': pc1_pre, 'out2': out2.detach().squeeze(2).cpu(),
                'target2': occupancy2.detach().cpu(), 'samples2': samples2,
                'partial': partial.detach().cpu(), 'ground_truth': ground_truth}

    def validation_step_end(self, output):
        pred, trgt = output['out2'], output['target2'].int()

        self.accuracy(pred, trgt), self.precision_(pred, trgt)
        self.avg_chamfer(chamfer_batch_pc(output['pc1'], output['ground_truth']))
        self.pre_adpt_chamfer(chamfer_batch_pc(output['pre_pc'], output['ground_truth']))
        self.recall(pred, trgt), self.f1(pred, trgt), self.avg_loss(F.binary_cross_entropy(pred, trgt.float()))

        return output

    def validation_epoch_end(self, output):
        self.log('valid/accuracy', self.accuracy.compute())
        self.log('valid/precision', self.precision_.compute())
        self.log('valid/recall', self.recall.compute())
        self.log('valid/f1', self.f1.compute())
        self.log('valid/chamfer', self.avg_chamfer.compute())
        self.log('valid/loss', self.avg_loss.compute())
        self.log('valid/chamfer_pre', self.pre_adpt_chamfer.compute())

        idxs = [np.random.randint(0, len(output)), -1]
        # print(f'idxs={idxs}')
        # print(f'len(output)={len(output)}')

        for idx, name in zip(idxs, ['random', 'fixed']):
            batch = output[idx]

            out, trgt, samples = batch['out2'][-1], batch['target2'][-1], batch['samples2'][-1]

            pred = out > 0.5

            # all positive predictions with labels for true positive and false positives
            colors = torch.zeros_like(samples, device='cpu')
            colors[trgt.bool().squeeze()] = torch.tensor([0, 255., 0])
            colors[~ trgt.bool().squeeze()] = torch.tensor([255., 0, 0])

            precision_pc = torch.cat((samples.cpu(), colors), dim=-1).detach().cpu().numpy()
            precision_pc = precision_pc[pred.squeeze()]

            # all true points with labels for true positive and false negatives
            colors = torch.zeros_like(samples, device='cpu')
            colors[pred.squeeze()] = torch.tensor([0, 255., 0])
            colors[~ pred.squeeze()] = torch.tensor([255., 0, 0])

            recall_pc = torch.cat((samples.cpu(), colors), dim=-1).detach().cpu().numpy()
            recall_pc = recall_pc[(trgt == 1.).squeeze()]


            complete = batch['ground_truth'][-1].detach().cpu().numpy()

            partial = torch.cat([batch['partial'][-1], torch.tensor([[255., 165. , 0.]]).tile(batch['partial'][-1].shape[0], 1)],
                      dim=-1).detach().cpu().numpy()

            reconstruction = batch['pc1'][-1]
            idxs = (reconstruction[..., 0] > 0.5) + (reconstruction[..., 0] < -0.5) + (reconstruction[..., 1] > 0.5) + (reconstruction[..., 1] < -0.5) + (
                    reconstruction[..., 2] > 0.5) + (reconstruction[..., 2] < -0.5)
            reconstruction = reconstruction[~idxs]
            reconstruction = torch.cat([reconstruction.detach().cpu(), torch.tensor([[0., 0., 255.]]).tile(reconstruction.shape[0], 1)], dim=-1).numpy()

            # TODO Fix partial and Add colors
            if Config.Eval.wandb:
                self.trainer.logger.experiment[0].log(
                    {f'{name}_precision_pc': wandb.Object3D({"points": precision_pc, 'type': 'lidar/beta'})})
                self.trainer.logger.experiment[0].log(
                    {f'{name}_recall_pc': wandb.Object3D({"points": recall_pc, 'type': 'lidar/beta'})})
                self.trainer.logger.experiment[0].log(
                    {f'{name}_partial_pc': wandb.Object3D({"points": partial, 'type': 'lidar/beta'})})
                self.trainer.logger.experiment[0].log(
                    {f'{name}_complete_pc': wandb.Object3D({"points": complete, 'type': 'lidar/beta'})})
                self.trainer.logger.experiment[0].log(
                    {f'{name}_reconstruction': wandb.Object3D({"points": np.concatenate([reconstruction, partial], axis=0), 'type': 'lidar/beta'})})
