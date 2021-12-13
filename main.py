import os
import time
from torch.optim import SGD

from utils.lightning import SplitProgressBar

try:
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cuda.pybind.visualization import draw_geometries
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cpu.pybind.visualization import draw_geometries
    from open3d.cpu.pybind.geometry import PointCloud

from configs.server_config import DataConfig, ModelConfig, TrainConfig

os.environ['CUDA_VISIBLE_DEVICES'] = TrainConfig.visible_dev
from pytorch_lightning.callbacks import GPUStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, Precision, Recall, F1, AverageMeter
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.ShapeNetPOVRemoval import BoxNet
from models.HyperNetwork import BackBone, ImplicitFunction
import torch
import copy
from utils.misc import create_3d_grid, check_mesh_contains, create_cube
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

        # self.apply(self._init_weights)
        for parameter in self.backbone.transformer.parameters():
            if len(parameter.size()) > 2:
                torch.nn.init.xavier_uniform_(parameter)

        self.accuracy = Accuracy()
        self.precision_ = Precision()
        self.recall = Recall()
        self.f1 = F1()
        self.avg_loss = AverageMeter()
        self.avg_chamfer = AverageMeter()

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
        self.training_set = BoxNet(DataConfig, DataConfig.train_samples)

        self.valid_set = BoxNet(DataConfig, DataConfig.val_samples)

    def train_dataloader(self):
        dl = DataLoader(self.training_set,
                        batch_size=TrainConfig.mb_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=TrainConfig.num_workers,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=generator)

        return dl

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            shuffle=False,
            batch_size=TrainConfig.test_mb_size,
            drop_last=False,
            num_workers=TrainConfig.num_workers,
            pin_memory=True)

    def forward(self, partial, object_id=None, step=0.04):
        samples = create_3d_grid(batch_size=partial.shape[0], step=step).to(TrainConfig.device)

        # start = time.time()
        fast_weights, _ = self.backbone(partial)
        # print("Create Fast Weights: {}".format(time.time() - start))

        prediction = torch.sigmoid(self.sdf(samples, fast_weights))

        return prediction, fast_weights, samples

    def configure_optimizers(self):
        optimizer = TrainConfig.optimizer(self.parameters(), lr=TrainConfig.lr, weight_decay=TrainConfig.wd)
        return optimizer

    def on_train_start(self) -> None:
        wandb.watch(self, log='all', log_freq=TrainConfig.log_metrics_every)

    def on_train_epoch_start(self):
        self.accuracy.reset(), self.precision_.reset(), self.recall.reset()
        self.f1.reset(), self.avg_loss.reset()

    def training_step(self, batch, batch_idx):
        label, partial, _, samples, occupancy = batch

        one_hot = None
        if ModelConfig.use_object_id:
            one_hot = torch.zeros((label.shape[0], DataConfig.n_classes), dtype=torch.float).to(label.device)
            one_hot[torch.arange(0, label.shape[0]), label] = 1.

        fast_weights, _ = self.backbone(partial, object_id=one_hot)

        ### Adaptation
        adaptation_steps = 10

        fast_weights = [[t.clone().detach().requires_grad_(True) for t in l] for l in fast_weights]
        optim = SGD(sum(fast_weights, []), lr=0.5, momentum=0.9)
        adpt_samples = partial # should add more negarive samples (e.g. all the one on the same lines of positive samples)
        for _ in range(adaptation_steps):
            out = self.sdf(adpt_samples, fast_weights)
            loss = F.binary_cross_entropy_with_logits(out, torch.ones(partial.shape[:2] + (1,), device=TrainConfig.device))

            optim.zero_grad()
            loss.backward(inputs=sum(fast_weights, []))
            optim.step()

        ### Adaptation

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

    def on_validation_epoch_start(self):
        self.accuracy.reset(), self.precision_.reset(), self.recall.reset()
        self.f1.reset(), self.avg_loss.reset(), self.avg_chamfer.reset()

    def validation_step(self, batch, batch_idx):
        label, partial, meshes, _, _ = batch

        verts, tris = meshes
        meshes_list = []
        for vert, tri in zip(verts, tris):
            meshes_list.append(o3d.geometry.TriangleMesh(Vector3dVector(vert.cpu()), Vector3iVector(tri.cpu())))

        self.grid = create_3d_grid(batch_size=label.shape[0],
                                   step=TrainConfig.grid_res_step).to(TrainConfig.device)

        occupancy = check_mesh_contains(meshes_list, self.grid, max_dist=0.01)  # TODO PARALLELIZE IT
        # Andrea said         occupancy = check_mesh_contains(meshes_list, self.grid.cpu().tolist(), tolerance=0.01).tolist()
        occupancy = torch.FloatTensor(occupancy).to(TrainConfig.device)

        one_hot = None
        if ModelConfig.use_object_id:
            one_hot = torch.zeros((label.shape[0], DataConfig.n_classes), dtype=torch.float).to(label.device)
            one_hot[torch.arange(0, label.shape[0]), label] = 1.

        fast_weights, _ = self.backbone(partial, object_id=one_hot)
        out = self.sdf(self.grid, fast_weights)

        return {'out': torch.sigmoid(out).detach().cpu(), 'target': occupancy.detach().cpu(),
                'mesh': meshes_list, 'partial': partial.detach().cpu(), 'label': label.detach().cpu()}

    def validation_step_end(self, output):
        pred, trgt, mesh = output['out'], output['target'].int(), output['mesh']
        self.accuracy(pred, trgt), self.precision_(pred, trgt), self.avg_chamfer(chamfer(self.grid, pred, mesh))
        self.recall(pred, trgt), self.f1(pred, trgt), self.avg_loss(F.binary_cross_entropy(pred, trgt.float()))

        return output

    def validation_epoch_end(self, output):
        self.log('valid/accuracy', self.accuracy.compute())
        self.log('valid/precision', self.precision_.compute())
        self.log('valid/recall', self.recall.compute())
        self.log('valid/f1', self.f1.compute())
        self.log('valid/chamfer', self.avg_chamfer.compute())
        self.log('valid/loss', self.avg_loss.compute())

        idxs = [np.random.randint(0, len(output)), -1]

        for idx, name in zip(idxs, ['random', 'fixed']):
            batch = output[idx]
            out, trgt, mesh = batch['out'][-1], batch['target'][-1], batch['mesh'][-1]
            pred = out > 0.5

            # all positive predictions with labels for true positive and false positives
            colors = torch.zeros_like(self.grid[0], device='cpu')
            colors[trgt.bool().squeeze()] = torch.tensor([0, 255., 0])
            colors[~ trgt.bool().squeeze()] = torch.tensor([255., 0, 0])

            precision_pc = torch.cat((self.grid[0].cpu(), colors), dim=-1).detach().cpu().numpy()
            precision_pc = precision_pc[pred.squeeze()]

            # all true points with labels for true positive and false negatives
            colors = torch.zeros_like(self.grid[0], device='cpu')
            colors[pred.squeeze()] = torch.tensor([0, 255., 0])
            colors[~ pred.squeeze()] = torch.tensor([255., 0, 0])

            recall_pc = torch.cat((self.grid[0].cpu(), colors), dim=-1).detach().cpu().numpy()
            recall_pc = recall_pc[(trgt == 1.).squeeze()]

            # complete = o3d.io.read_triangle_mesh(mesh[-1], False)
            complete = mesh

            complete = complete.sample_points_uniformly(10000)
            complete = np.array(complete.points)

            partial = batch['partial'][-1]
            partial = np.array(partial.squeeze())

            # TODO Fix partial and Add colors
            # pc = PointCloud()
            # pc.points = Vector3dVector(partial)
            # o3d.visualization.draw_geometries([mesh, pc], window_name="Partial")

            self.trainer.logger.experiment[0].log(
                {f'{name}_precision_pc': wandb.Object3D({"points": precision_pc, 'type': 'lidar/beta'})})
            self.trainer.logger.experiment[0].log(
                {f'{name}_recall_pc': wandb.Object3D({"points": recall_pc, 'type': 'lidar/beta'})})
            self.trainer.logger.experiment[0].log(
                {f'{name}_partial_pc': wandb.Object3D({"points": partial, 'type': 'lidar/beta'})})
            self.trainer.logger.experiment[0].log(
                {f'{name}_complete_pc': wandb.Object3D({"points": complete, 'type': 'lidar/beta'})})
            pass


def visualize(meshes, partials):
    mesh_paths, rotations, means, vars = meshes
    p, r, m, v = mesh_paths[0], rotations[0], means[0], vars[0]

    mesh1 = o3d.io.read_triangle_mesh(p, False)
    mesh2 = copy.deepcopy(mesh1)
    mesh1.rotate(r.cpu().numpy())
    mesh1.translate(-m.cpu().numpy())
    mesh1.scale(1 / (v.cpu().numpy() * 2), center=[0, 0, 0])

    pc = PointCloud()
    pc.points = Vector3dVector(partials[0].cpu().numpy())

    draw_geometries([mesh1, mesh2, pc, create_cube()])


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
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh)
        query_points = o3d.core.Tensor(query.cpu().numpy(), dtype=o3d.core.Dtype.Float32)
        signed_distance = scene.compute_distance(query_points)
        return np.mean(signed_distance.numpy())


if __name__ == '__main__':
    print(TrainConfig.num_workers)

    model = HyperNetwork(ModelConfig)

    config = {'train': {k: dict(TrainConfig.__dict__)[k] for k in dict(TrainConfig.__dict__) if
                        not k.startswith("__")},
              'model': {k: dict(ModelConfig.__dict__)[k] for k in dict(ModelConfig.__dict__) if
                        not k.startswith("__")},
              'data': {k: dict(DataConfig.__dict__)[k] for k in dict(DataConfig.__dict__) if
                       not k.startswith("__")}}

    wandb.login(key="f5f77cf17fad38aaa2db860576eee24bde163b7a")
    wandb.init(project="train_box")
    wandb_logger = WandbLogger(project='pcr', log_model='all', config=config)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        mode='max',
        auto_insert_metric_name=False)

    trainer = pl.Trainer(max_epochs=TrainConfig.n_epoch,
                         precision=32,
                         gpus=1,
                         log_every_n_steps=TrainConfig.log_metrics_every,
                         logger=[wandb_logger],
                         gradient_clip_val=TrainConfig.clip_value,
                         gradient_clip_algorithm='value',
                         callbacks=[GPUStatsMonitor(),
                                    SplitProgressBar(),
                                    checkpoint_callback],
                         )

    trainer.fit(model)
