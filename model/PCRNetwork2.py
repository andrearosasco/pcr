import copy
from abc import ABC
from collections import defaultdict
import numpy as np
import torch
from datasets.GraspingDataset import GraspingDataset
from utils.metrics import chamfer_batch_pc
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

from configs import Config

from torchmetrics import Accuracy, Precision, Recall, F1Score, MeanMetric
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.ShapeNetPOVDepth import ShapeNet as Dataset

from utils.misc import create_3d_grid, sample_point_cloud_pc
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


        m = {'accuracy': Accuracy(),
            'precision': Precision(),
            'recall': Recall(),
            'f1': F1Score(),
            'loss': MeanMetric(),
            'chamfer': MeanMetric()}

        self.metrics = {
            'train': m,
            'val_models': {**copy.deepcopy(m), **{'chamfer': MeanMetric()}},
            'val_views': {**copy.deepcopy(m), **{'chamfer': MeanMetric()}}
        }

        self.training_set, self.valid_set_models, self.valid_set_views = None, None, None

        self.rt_setup = False

        self.cls_count = defaultdict(lambda: 0)
        self.cls_cd = defaultdict(lambda: 0)

        self.cd = []
        self.labels = []

        self.val_outputs = None

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
        root = Config.Data.dataset_path
        split = 'data/MCD/build_datasets/train_test_dataset.json'

        self.training_set = GraspingDataset(root, split, subset='train_models_train_views')
        self.valid_set_models = GraspingDataset(root, split, subset='holdout_models_holdout_views')
        self.valid_set_views = GraspingDataset(root, split, subset='train_models_holdout_views')

    def train_dataloader(self):
        dl = DataLoader(self.training_set,
                        batch_size=Config.Train.mb_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=Config.General.num_workers,
                        pin_memory=True)

        return dl

    def val_dataloader(self):
        # Smaller one
        dl1 = DataLoader(
            self.valid_set_views,
            shuffle=False,
            batch_size=Config.Eval.mb_size,
            drop_last=False,
            num_workers=Config.General.num_workers,
            pin_memory=True)

        #  Bigger one
        dl2 = DataLoader(
            self.valid_set_models,
            shuffle=False,
            batch_size=Config.Eval.mb_size,
            drop_last=False,
            num_workers=Config.General.num_workers,
            pin_memory=True)

        return [dl1, dl2]

    def forward(self, partial, object_id=None, step=0.01):
        samples = create_3d_grid(batch_size=partial.shape[0], step=step).to(Config.General.device)

        fast_weights, _ = self.backbone(partial)
        prediction = torch.sigmoid(self.sdf(samples, fast_weights))

        return samples[prediction.squeeze(-1) > 0.5], fast_weights

    def configure_optimizers(self):
        optimizer = Config.Train.optimizer(self.parameters(), lr=Config.Train.lr, weight_decay=Config.Train.wd)
        return optimizer

    def on_train_epoch_start(self):
        for m in self.metrics['train'].values():
            m.reset()

    def training_step(self, batch, batch_idx):
        partial, ground_truth = batch

        fast_weights, _ = self.backbone(partial)
        ### Adaptation
        if Config.Train.adaptation:
            adaptation_steps = 10

            fast_weights = [[t.requires_grad_(True) for t in l] for l in fast_weights]

            for _ in range(adaptation_steps):
                optim = DifferentiableSGD(sum(fast_weights, []), lr=0.1,
                                          momentum=0.9)  # the sum flatten the list of list

                out = self.sdf(partial, fast_weights)
                # The loss function also computes the sigmoid
                loss = F.binary_cross_entropy_with_logits(out, torch.ones(partial.shape[:2] + (1,),
                                                                          device=Config.General.device))

                loss.backward(inputs=sum(fast_weights, []), retain_graph=True)
                fast_weights = optim.step()
                fast_weights = [[fast_weights[i].to(Config.General.device),
                                 fast_weights[i + 1].to(Config.General.device),
                                 fast_weights[i + 2].to(Config.General.device)]
                                for i in range(0, 3 * (Config.Model.depth + 2), 3)]

        samples, target = sample_point_cloud_pc(ground_truth, n_points=Config.Data.implicit_input_dimension,
                                                dist=Config.Data.dist,
                                                noise_rate=Config.Data.noise_rate,
                                                tolerance=Config.Data.tolerance)
        target = target.float()

        out = self.sdf(samples, fast_weights)
        pred = torch.sigmoid(out.detach()).cpu()

        loss = F.binary_cross_entropy_with_logits(out, target.unsqueeze(-1))
        target = target.unsqueeze(-1).int()

        return {'loss': loss, 'pred': pred.detach().cpu(), 'target': target.detach().cpu()}

    @torch.no_grad()
    def training_step_end(self, output):
        pred, trgt = output['pred'], output['target']
        pred = torch.nan_to_num(pred)

        # This log the metrics on the current batch and accumulate it in the average
        metrics = self.metrics['train']


        self.log('train/accuracy', metrics['accuracy'](pred, trgt))
        self.log('train/precision', metrics['precision'](pred, trgt))
        self.log('train/recall', metrics['recall'](pred, trgt))
        self.log('train/f1', metrics['f1'](pred, trgt))

        self.log('train/loss', metrics['loss'](torch.nan_to_num(output['loss'].detach().cpu(), nan=1)))

    def on_validation_epoch_start(self):
        self.random_batch = np.random.randint(0, int(len(self.valid_set_models) / Config.Data.Eval.mb_size) - 1)

        for m in self.metrics['val_models'].values():
            m.reset()
        for m in self.metrics['val_views'].values():
            m.reset()

    def validation_step(self, batch, batch_idx, dl_idx):
        partial, ground_truth = batch

        # We always validate on the small valid_set

        # We validate on the big valid_set once every 10 epochs
        # Still we log a random and a fixed reconstruction from the
        # big dataset every epoch.
        if dl_idx == 1:
            # Sample one fixed and one random batch for point cloud logging

            if self.random_batch == batch_idx:
                samples2, occupancy2 = sample_point_cloud_pc(ground_truth,
                                                             n_points=Config.Data.implicit_input_dimension,
                                                             dist=Config.Data.dist,
                                                             noise_rate=Config.Data.noise_rate,
                                                             tolerance=Config.Data.tolerance)
                fast_weights, _ = self.backbone(partial)
                pc1 = self.decoder(fast_weights)
                out2 = torch.sigmoid(self.sdf(samples2, fast_weights))
                self.val_outputs = {'random': {'pc1': pc1, 'out2': out2.detach().squeeze(2).cpu(),
                                               'target2': occupancy2.detach().cpu(),
                                               'samples2': samples2.detach().cpu(),
                                               'partial': partial.detach().cpu(), 'ground_truth': ground_truth,
                                               'batch_idx': batch_idx, 'dl_idx': dl_idx}
                                    }

            if int(len(self.valid_set_models) / Config.Data.Eval.mb_size) - 2 == batch_idx:
                samples2, occupancy2 = sample_point_cloud_pc(ground_truth,
                                                             n_points=Config.Data.implicit_input_dimension,
                                                             dist=Config.Data.dist,
                                                             noise_rate=Config.Data.noise_rate,
                                                             tolerance=Config.Data.tolerance)
                fast_weights, _ = self.backbone(partial)
                pc1 = self.decoder(fast_weights)
                out2 = torch.sigmoid(self.sdf(samples2, fast_weights))
                self.val_outputs['fixed'] = {'pc1': pc1, 'out2': out2.detach().squeeze(2).cpu(),
                                             'target2': occupancy2.detach().cpu(), 'samples2': samples2.detach().cpu(),
                                             'partial': partial.detach().cpu(), 'ground_truth': ground_truth,
                                             'batch_idx': batch_idx, 'dl_idx': dl_idx}

            if (self.current_epoch + 1) % 10 != 0:
                return {'dl_idx': dl_idx}


        # The sampling with "sample_point_cloud" simulate the sampling used during training
        # This is useful as we always get ground truths sampling on the meshes but it doesn't reflect
        #   how the algorithm will work after deployment

        samples2, occupancy2 = sample_point_cloud_pc(ground_truth, n_points=Config.Data.implicit_input_dimension,
                                                     dist=Config.Data.dist,
                                                     noise_rate=Config.Data.noise_rate, tolerance=Config.Data.tolerance)

        ############# INFERENCE #############
        fast_weights, _ = self.backbone(partial)

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
                    fast_weights = [[fast_weights[i], fast_weights[i + 1], fast_weights[i + 2]] for i in
                                    range(0, 3 * (Config.Model.depth + 2), 3)]

        pc1 = self.decoder(fast_weights)
        out2 = torch.sigmoid(self.sdf(samples2, fast_weights))

        return {'pc1': pc1, 'out2': out2.detach().squeeze(2).cpu(),
                'target2': occupancy2.detach().cpu(), 'samples2': samples2.detach().cpu(),
                'partial': partial.detach().cpu(), 'ground_truth': ground_truth, 'batch_idx': batch_idx, 'dl_idx': dl_idx}

    def validation_step_end(self, output):
        if (self.current_epoch + 1) % 10 != 0:
            if output['dl_idx'] == 1:
                self.trainer._active_loop.outputs = []
                return

        pred, trgt = output['out2'], output['target2'].int()
        pred = torch.nan_to_num(pred)

        if output['dl_idx'] == 0:
            metrics = self.metrics['val_views']
        if output['dl_idx'] == 1:
            metrics = self.metrics['val_models']

        metrics['accuracy'](pred, trgt), metrics['precision'](pred, trgt)
        metrics['chamfer'](chamfer_batch_pc(output['pc1'], output['ground_truth']).cpu())
        metrics['recall'](pred, trgt), metrics['f1'](pred, trgt)
        metrics['loss'](F.binary_cross_entropy(pred, trgt.float()))

        output['pc1'], output['ground_truth'] = output['pc1'].detach().cpu(), output['ground_truth'].detach().cpu()

        self.trainer._active_loop.outputs = []

    def validation_epoch_end(self, output):

        for k, m in self.metrics['val_views'].items():
            self.log(f'val_views/{k}', m.compute())

        if (self.current_epoch + 1) % 10 == 0:
            for k, m in self.metrics['val_models'].items():
                self.log(f'val_models/{k}', m.compute())

        if self.val_outputs is None:
            return

        for name, batch in self.val_outputs.items():

            out, trgt, samples = batch['out2'][-1], batch['target2'][-1], batch['samples2'][-1]

            pred = out > 0.5

            # all positive predictions with labels for true positive and false positives
            colors = torch.zeros_like(samples, device='cpu')
            colors[trgt.bool().squeeze()] = torch.tensor([0, 255., 0])
            colors[~trgt.bool().squeeze()] = torch.tensor([255., 0, 0])

            precision_pc = torch.cat((samples.cpu(), colors), dim=-1).detach().cpu().numpy()
            precision_pc = precision_pc[pred.squeeze()]

            # all true points with labels for true positive and false negatives
            colors = torch.zeros_like(samples, device='cpu')
            colors[pred.squeeze()] = torch.tensor([0, 255., 0])
            colors[~ pred.squeeze()] = torch.tensor([255., 0, 0])

            recall_pc = torch.cat((samples.cpu(), colors), dim=-1).detach().cpu().numpy()
            recall_pc = recall_pc[(trgt == 1.).squeeze()]

            complete = batch['ground_truth'][-1].detach().cpu().numpy()

            partial = torch.cat(
                [batch['partial'][-1], torch.tensor([[255., 165., 0.]]).tile(batch['partial'][-1].shape[0], 1)],
                dim=-1).detach().cpu().numpy()

            reconstruction = batch['pc1'][-1]
            idxs = (reconstruction[..., 0] > 0.5) + (reconstruction[..., 0] < -0.5) + (reconstruction[..., 1] > 0.5) + (
                        reconstruction[..., 1] < -0.5) + (
                           reconstruction[..., 2] > 0.5) + (reconstruction[..., 2] < -0.5)
            reconstruction = reconstruction[~idxs]
            reconstruction = torch.cat(
                [reconstruction.detach().cpu(), torch.tensor([[0., 0., 255.]]).tile(reconstruction.shape[0], 1)],
                dim=-1).numpy()

            # TODO Fix partial and Add colors
            if Config.Eval.wandb:
                self.trainer.logger.experiment.log(
                    {f'{name}_precision_pc': wandb.Object3D({"points": precision_pc, 'type': 'lidar/beta'})})
                self.trainer.logger.experiment.log(
                    {f'{name}_recall_pc': wandb.Object3D({"points": recall_pc, 'type': 'lidar/beta'})})
                self.trainer.logger.experiment.log(
                    {f'{name}_partial_pc': wandb.Object3D({"points": partial, 'type': 'lidar/beta'})})
                self.trainer.logger.experiment.log(
                    {f'{name}_complete_pc': wandb.Object3D({"points": complete, 'type': 'lidar/beta'})})
                self.trainer.logger.experiment.log(
                    {f'{name}_reconstruction': wandb.Object3D(
                        {"points": np.concatenate([reconstruction, partial], axis=0), 'type': 'lidar/beta'})})
