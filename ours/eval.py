import math
import os
import sys

from configs import DataConfig, ModelConfig, TrainConfig, server_config

os.environ['CUDA_VISIBLE_DEVICES'] = TrainConfig.visible_dev
from pytorch_lightning.callbacks import GPUStatsMonitor, ProgressBar, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, Precision, Recall, F1, AverageMeter

import random
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from datasets.ShapeNetPOV import ShapeNet
from models.HyperNetwork import BackBone, ImplicitFunction
import torch
from tqdm import tqdm
import copy
from utils.misc import create_3d_grid, check_mesh_contains
import open3d as o3d
import wandb

import pytorch_lightning as pl


# =======================  Reproducibility =======================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
seed = TrainConfig.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
generator = torch.Generator()
generator.manual_seed(TrainConfig.seed)


# ================================================================


class HyperNetwork(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.backbone = BackBone(config)
        self.sdf = ImplicitFunction(config)

        for parameter in self.backbone.transformer.parameters():
            if len(parameter.size()) > 2:
                torch.nn.init.xavier_uniform_(parameter)

        self.accuracy = Accuracy()
        self.precision_ = Precision()
        self.recall = Recall()
        self.f1 = F1()
        self.avg_loss = AverageMeter()
        self.avg_chamfer = AverageMeter()

    def prepare_data(self):
        self.training_set = ShapeNet(DataConfig,
                                     mode=f"{DataConfig.mode}/train",
                                     overfit_mode=TrainConfig.overfit_mode)

        self.valid_set = ShapeNet(DataConfig,
                                  mode=f"{DataConfig.mode}/valid",
                                  overfit_mode=TrainConfig.overfit_mode)

    def train_dataloader(self):
        dl = DataLoader(self.training_set,
                          batch_size=TrainConfig.mb_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=TrainConfig.num_workers,
                          pin_memory=True,
                          worker_init_fn=seed_worker,
                          generator=generator)

        print(len(self.training_set), '/', TrainConfig.mb_size, '=', len(dl))
        return dl

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=TrainConfig.test_mb_size,
            drop_last=True,
            num_workers=TrainConfig.num_workers,
            pin_memory=True)

    def forward(self, partial, object_id=None):
        samples = create_3d_grid(batch_size=partial.shape[0]).to(TrainConfig.device)

        if object_id is not None:
            one_hot = torch.zeros((partial.shape[0], DataConfig.n_classes), dtype=torch.float).to(partial.device)
            one_hot[torch.arange(0, partial.shape[0]), object_id] = 1.

        fast_weights, _ = self.backbone(partial, object_id=one_hot)
        prediction = torch.sigmoid(self.sdf(samples, fast_weights))

        return prediction

    def configure_optimizers(self):
        optimizer = TrainConfig.optimizer(self.parameters(), lr=TrainConfig.lr)
        return optimizer

    def on_train_epoch_start(self):
        self.accuracy.reset(), self.precision_.reset(), self.recall.reset()
        self.f1.reset(), self.avg_loss.reset()

    def training_step(self, batch, batch_idx):
        label, partial, complete, samples, occupancy, _ = batch

        one_hot = None
        if ModelConfig.use_object_id:
            one_hot = torch.zeros((batch.shape[0], DataConfig.n_classes), dtype=torch.float).to(batch.device)
            one_hot[torch.arange(0, batch.shape[0]), label] = 1.

        fast_weights, _ = self.backbone(partial, object_id=one_hot)
        out = self.sdf(samples, fast_weights)

        loss = F.binary_cross_entropy_with_logits(out, occupancy.unsqueeze(-1))
        return {'loss': loss, 'out': out, 'target': occupancy}

    @torch.no_grad()
    def training_step_end(self, output):
        pred, trgt = torch.sigmoid(output['out']).detach().cpu(), output['target'].unsqueeze(-1).int().detach().cpu()
        self.accuracy(pred, trgt), self.precision_(pred, trgt)
        self.recall(pred, trgt), self.f1(pred, trgt), self.avg_loss(output['loss'].detach().cpu())

        self.log('Performances', {'train/accuracy': self.accuracy, 'train/precision': self.precision_,
                                  'train/recall': self.recall, 'train/f1': self.f1,
                                  'train/loss': self.avg_loss, 'train/step': self.global_step})

    def on_validation_epoch_start(self):
        self.accuracy.reset(), self.precision_.reset(), self.recall.reset()
        self.f1.reset(), self.avg_loss.reset(), self.avg_chamfer.reset()

        self.grid = create_3d_grid(batch_size=TrainConfig.test_mb_size,
                                   step=TrainConfig.grid_res_step).to(TrainConfig.device)

        self.current_label = 0

    def validation_step(self, batch, batch_idx):
        label, partial, mesh = batch

        if int(label.item()) != self.current_label:
            self.current_label = int(label.item())
            self.accuracy.reset(), self.precision_.reset(), self.recall.reset()
            self.f1.reset(), self.avg_loss.reset(), self.avg_chamfer.reset()


        occupancy = check_mesh_contains(mesh, self.grid, max_dist=0.01)  # TODO PARALLELIZE IT
        occupancy = torch.FloatTensor(occupancy).to(TrainConfig.device)

        one_hot = None
        if ModelConfig.use_object_id:
            one_hot = torch.zeros((batch.shape[0], DataConfig.n_classes), dtype=torch.float).to(batch.device)
            one_hot[torch.arange(0, batch.shape[0]), label] = 1.

        fast_weights, _ = self.backbone(partial, object_id=one_hot)
        out = self.sdf(self.grid, fast_weights)

        return {'out': torch.sigmoid(out).detach().cpu(), 'target': occupancy.detach().cpu(),
                'mesh': mesh, 'partial': partial.detach().cpu(), 'label': label.detach().cpu()}

    def validation_step_end(self, output):
        pred, trgt, mesh = output['out'], output['target'].int(), output['mesh']
        self.accuracy.update(pred, trgt), self.precision_.update(pred, trgt), self.avg_chamfer.update(chamfer(self.grid, pred, mesh))
        self.recall.update(pred, trgt), self.f1.update(pred, trgt), self.avg_loss.update(F.binary_cross_entropy(pred, trgt.float()))

        l = output['label'].item()

        self.log(f'valid/precision_{l}', self.precision_.compute(), on_step=True)
        self.log(f'valid/recall_{l}', self.recall.compute(), on_step=True)
        self.log(f'valid/f1_{l}', self.f1.compute(), on_step=True)
        self.log(f'valid/chamfer_{l}', self.avg_chamfer.compute(), on_step=True)
        self.log(f'valid/loss_{l}', self.avg_loss.compute(), on_step=True)
        return output

    # def validation_epoch_end(self, output):
    #     self.log('valid/accuracy', self.accuracy)
    #     self.log('valid/precision', self.precision_)
    #     self.log('valid/recall', self.recall)
    #     self.log('valid/f1', self.f1)
    #     self.log('valid/chamfer', self.avg_chamfer)
    #     self.log('valid/loss', self.avg_loss)

        # output = output[0]
        # pred, trgt, mesh = output['out'][0], output['target'][0], output['mesh'][0]
        #
        # # all positive predictions with labels for true positive and false positives
        # precision_pc = torch.cat((self.grid[0], trgt), dim=-1).detach().cpu().numpy()
        # precision_pc = precision_pc[(pred.cpu().numpy() == 1).squeeze()]
        #
        # # all true points with labels for true positive and false negatives
        # recall_pc = torch.cat((self.grid[0], pred), dim=-1).detach().cpu().numpy()
        # recall_pc = recall_pc[(trgt.cpu().numpy() == 1.).squeeze()]
        #
        # complete = o3d.io.read_triangle_mesh(mesh, False)
        #
        # complete = complete.sample_points_uniformly(10000)
        # complete = np.array(complete.points)
        #
        # partial = output['partial']
        #
        # self.trainer.logger.experiment[0].log({
        #     'precision_pc': wandb.Object3D({"points": precision_pc, 'type': 'lidar/beta'}),
        #     'recall_pc': wandb.Object3D({"points": recall_pc, 'type': 'lidar/beta'}),
        #     'partial_pc': wandb.Object3D({"points": partial, 'type': 'lidar/beta'}),
        #     'complete_pc': wandb.Object3D({"points": complete, 'type': 'lidar/beta'})
        # })


def print_memory():
    import gc
    i = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    i += obj.reshape(-1, 1).shape[0]
        except:
            pass

    print(i)


def chamfer(samples, predictions, meshes):
    for mesh, pred in zip(meshes, predictions):
        query = samples[0, (pred > 0.5).squeeze()]

        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.io.read_triangle_mesh(mesh, False)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh)
        query_points = o3d.core.Tensor(query.cpu().numpy(), dtype=o3d.core.Dtype.Float32)
        signed_distance = scene.compute_distance(query_points)
        return np.mean(signed_distance.numpy())


if __name__ == '__main__':
    # print_memory()
    model = HyperNetwork(ModelConfig)
    # model.to('cuda')
    # print_memory()

    wandb_logger = WandbLogger(project='pcr', log_model='all', entity='coredump')
    wandb_logger.watch(model)

    # checkpoint = torch.load('./checkpoint/20-10-21_0732.ptc')
    # aux = {}
    # aux['state_dict'] = checkpoint
    # torch.save(aux, './checkpoint/best.ptc')
    model = HyperNetwork.load_from_checkpoint('./checkpoint/best.ptc', config=server_config.ModelConfig,)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        auto_insert_metric_name=False)

    class LitProgressBar(ProgressBar):

        def on_train_epoch_start(self, trainer, pl_module):
            super().on_train_epoch_start(trainer, pl_module)
            total_train_batches = self.total_train_batches

            total_batches = total_train_batches
            if total_batches is None or math.isinf(total_batches) or math.isnan(total_batches):
                total_batches = None
            if not self.main_progress_bar.disable:
                self.main_progress_bar.reset(total=total_batches)
            self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")


    bar = LitProgressBar()

    trainer = pl.Trainer(max_epochs=TrainConfig.n_epoch,
                         precision=32,
                         gpus=1,
                         log_every_n_steps=TrainConfig.log_metrics_every,
                         logger=[wandb_logger],
                         gradient_clip_val=TrainConfig.clip_value,
                         gradient_clip_algorithm='value',
                         callbacks=[GPUStatsMonitor(),
                                    bar,
                                    checkpoint_callback],
                         )

    # trainer.fit(model)
    trainer.validate(model)
