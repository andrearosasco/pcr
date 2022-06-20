from abc import ABC
from pathlib import Path

import numpy as np
import torch

from utils.metrics import chamfer_batch
from utils.reproducibility import get_init_fn, get_generator
from utils.sgdiff import DifferentiableSGD
from .Backbone import BackBone
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

from utils.misc import create_3d_grid, check_mesh_contains, sample_point_cloud, read_mesh_debug, read_mesh
import open3d as o3d
import pytorch_lightning as pl
import wandb


class PCRNetwork(pl.LightningModule, ABC):

    def __init__(self, config):
        super().__init__()
        self.backbone = BackBone(config)
        self.sdf = ImplicitFunction(config)
        self.decoder = Decoder(self.sdf)

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

        self.training_set, self.valid_set = None, None

        self.rt_setup = False

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
        self.training_set = Dataset(Config, mode=f"{Config.Data.mode}/train")
        self.valid_set = Dataset(Config, mode=f"{Config.Data.mode}/valid")

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
        label, partial, _, samples, occupancy = batch

        # for p, s, o in zip(partial.cpu().numpy(), samples.cpu().numpy(), occupancy.cpu().numpy()):
            # aux1 = PointCloud(points=Vector3dVector(p))
            # aux1.paint_uniform_color([0, 0, 1])
            # aux2 = PointCloud(points=Vector3dVector(s[o.astype(bool)]))
            # aux2.paint_uniform_color([0, 1, 0])
            # aux3 = PointCloud(points=Vector3dVector(s[~o.astype(bool)]))
            # aux3.paint_uniform_color([1, 0, 0])
            #
            # draw_geometries([aux1, aux2, aux3, o3d.geometry.TriangleMesh.create_coordinate_frame()])

        fast_weights, _ = self.backbone(partial)

        ### Adaptation
        if Config.Train.adaptation:
            adaptation_steps = 10

            fast_weights = [[t.requires_grad_(True) for t in l] for l in fast_weights]

            for _ in range(adaptation_steps):
                optim = DifferentiableSGD(sum(fast_weights, []), lr=0.1, momentum=0.9)  # the sum flatten the list of list

                out = self.sdf(partial, fast_weights)
                loss = F.binary_cross_entropy_with_logits(out, torch.ones(partial.shape[:2] + (1,), device=Config.General.device))

                loss.backward(inputs=sum(fast_weights, []), retain_graph=True)
                fast_weights = optim.step()
                fast_weights = [[fast_weights[i], fast_weights[i + 1], fast_weights[i + 2]] for i in range(0, 12, 3)]

        out = self.sdf(samples, fast_weights)

        loss = F.binary_cross_entropy_with_logits(out, occupancy.unsqueeze(-1))
        return {'loss': loss, 'out': out.detach().cpu(), 'target': occupancy.detach().cpu()}

    @torch.no_grad()
    def training_step_end(self, output):
        pred, trgt = torch.sigmoid(output['out']).detach().cpu(), output['target'].unsqueeze(-1).int().detach().cpu()

        # This log the metrics on the current batch and accumulate it in the average
        self.log('train/accuracy', self.accuracy(pred, trgt))
        self.log('train/precision', self.precision_(pred, trgt))
        self.log('train/recall', self.recall(pred, trgt))
        self.log('train/f1', self.f1(pred, trgt))
        self.log('train/loss', self.avg_loss(output['loss'].detach().cpu()))
        pass

    def on_validation_epoch_start(self):
        self.accuracy.reset(), self.precision_.reset(), self.recall.reset()
        self.f1.reset(), self.avg_loss.reset(), self.avg_chamfer.reset(), self.pre_adpt_chamfer.reset()

    def validation_step(self, batch, batch_idx):
        label, partial, meshes, _, _ = batch

        path, rotation, mean, var, offset = meshes

        meshes_list = []
        for p, r, m, s, o in zip(path, rotation, mean, var, offset):
            dir_path = Path(p)

            v = np.load(str(dir_path / 'models/model_vertices.npy'))
            t = np.load(str(dir_path / 'models/model_triangles.npy'))
            mesh = o3d.geometry.TriangleMesh(Vector3dVector(v),
                                             Vector3iVector(t))
            # mesh = read_mesh_debug(dir_path)

            mesh = mesh.rotate(r.cpu().numpy()).translate(-m.cpu().numpy())\
                .scale(1 / (s.cpu().numpy() * 2), center=[0, 0, 0]).translate(o.cpu().numpy())
            meshes_list.append(mesh)

        # The sampling with "sample_point_cloud" simulate the sampling used during training
        # This is useful as we always get ground truths sampling on the meshes but it doesn't reflect
        #   how the algorithm will work after deployment
        samples2, occupancy2 = [], []
        for mesh in meshes_list:
            s, o = sample_point_cloud(mesh, n_points=Config.Data.implicit_input_dimension, dist=Config.Data.dist,
                                      noise_rate=Config.Data.noise_rate, tolerance=Config.Data.tolerance)
            samples2.append(s), occupancy2.append(o)
        samples2 = torch.tensor(np.array(samples2)).float().to(Config.General.device)
        occupancy2 = torch.tensor(occupancy2, dtype=torch.float).unsqueeze(-1).to(Config.General.device)

        one_hot = None
        if Config.Model.use_object_id:
            one_hot = torch.zeros((label.shape[0], Config.Data.n_classes), dtype=torch.float).to(label.device)
            one_hot[torch.arange(0, label.shape[0]), label] = 1.

        ############# INFERENCE #############
        fast_weights, _ = self.backbone(partial, object_id=one_hot)
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

        return {'pc1': pc1, 'pre_pc': pc1_pre, 'out2': out2.detach().cpu(),
                'target2': occupancy2.detach().cpu(), 'samples2': samples2,
                'mesh': meshes_list, 'partial': partial.detach().cpu(), 'label': label.detach().cpu()}

    def validation_step_end(self, output):
        mesh = output['mesh']

        pred, trgt = output['out2'], output['target2'].int()

        self.accuracy(pred, trgt), self.precision_(pred, trgt)
        self.avg_chamfer(chamfer_batch(output['pc1'], mesh))
        self.pre_adpt_chamfer(chamfer_batch(output['pre_pc'], mesh))
        self.recall(pred, trgt), self.f1(pred, trgt), self.avg_loss(F.binary_cross_entropy(pred, trgt.float()))

        return output
        pass

    def validation_epoch_end(self, output):
        pass
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
            mesh = batch['mesh'][-1]

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

            complete = mesh

            complete = complete.sample_points_uniformly(10000)
            complete = np.array(complete.points)

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
