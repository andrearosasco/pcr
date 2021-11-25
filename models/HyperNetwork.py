import random
import numpy as np
import torch
import wandb
try:
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
except:
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import AverageMeter, F1, Recall, Precision, Accuracy
import open3d as o3d

from configs import DataConfig, ModelConfig, TrainConfig
from datasets.ShapeNetPOVRemoval import BoxNet
from utils.misc import check_mesh_contains, create_3d_grid, chamfer
from .Transformer import PCTransformer


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = (input_size + output_size)//2
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

generator = torch.Generator()
generator.manual_seed(TrainConfig.seed)

class HyperNetwork(LightningModule):

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
        self.training_set = BoxNet(DataConfig,
                                   DataConfig.train_samples)

        self.valid_set = BoxNet(DataConfig,
                                DataConfig.val_samples)

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

    def forward(self, partial, object_id=None):
        samples = create_3d_grid(batch_size=partial.shape[0]).to(TrainConfig.device)

        if object_id is not None:
            one_hot = torch.zeros((partial.shape[0], DataConfig.n_classes), dtype=torch.float).to(partial.device)
            one_hot[torch.arange(0, partial.shape[0]), object_id] = 1.

        fast_weights, _ = self.backbone(partial, object_id=one_hot)
        prediction = torch.sigmoid(self.sdf(samples, fast_weights))

        return prediction

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



class BackBone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.knn_layer = config.knn_layer

        self.transformer = PCTransformer(in_chans=config.n_channels,
                                         embed_dim=config.embed_dim,
                                         depth=config.encoder_depth,
                                         mlp_ratio=config.mlp_ratio,
                                         qkv_bias=config.qkv_bias,
                                         knn_layer=config.knn_layer,
                                         num_heads=config.num_heads,
                                         attn_drop_rate=config.attn_drop_rate,
                                         drop_rate=config.drop_rate,
                                         qk_scale=config.qk_scale,
                                         out_size=config.out_size)

        # Select between deep feature extractor and not
        if config.use_deep_weights_generator:
            generator = MLP
        else:
            generator = nn.Linear

        # Select the right dimension for linear layers
        global_size = config.out_size
        if config.use_object_id:
            global_size = config.out_size * 2

        # Generate first weight, bias and scale of the input layer of the implicit function
        self.output = nn.ModuleList([nn.ModuleList([
                generator(global_size, config.hidden_dim * 3),
                generator(global_size, config.hidden_dim),
                generator(global_size, config.hidden_dim)])])

        # Generate weights, biases and scales of the hidden layers of the implicit function
        for _ in range(config.depth):
            self.output.append(nn.ModuleList([
                    generator(global_size, config.hidden_dim * config.hidden_dim),
                    generator(global_size, config.hidden_dim),
                    generator(global_size, config.hidden_dim)
                ]))
        # Generate weights, biases and scales of the output layer of the implicit function
        self.output.append(nn.ModuleList([
            generator(global_size, config.hidden_dim),
            generator(global_size, 1),
            generator(global_size, 1),
        ]))

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=1)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, object_id=None):
        # xyz = torch.reshape(xyz, (xyz.shape[0], -1))
        # global_feature = self.test(xyz)

        global_feature = self.transformer(xyz)  # B M C and B M 3

        fast_weights = []
        for layer in self.output:
            fast_weights.append([ly(global_feature) for ly in layer])

        return fast_weights, global_feature


class ImplicitFunction(nn.Module):

    def __init__(self, config, params=None):
        super().__init__()
        self.params = params
        self.relu = nn.LeakyReLU(0.2)
        # self.dropout = nn.Dropout(0.5)
        self.hidden_dim = config.hidden_dim

    def set_params(self, params):
        self.params = params

    def forward(self, points, params=None):
        if params is not None:
            self.params = params

        if self.params is None:
            raise ValueError('Can not run forward on uninitialized implicit function')

        x = points
        # TODO: I just added unsqueeze(1), reshape(-1) and bmm and everything works (or did I introduce some kind of bug?)
        weights, scales, biases = self.params[0]
        weights = weights.reshape(-1, 3, self.hidden_dim)
        scales = scales.unsqueeze(1)
        biases = biases.unsqueeze(1)

        x = torch.bmm(x, weights) * scales + biases
        # x = self.dropout(x)
        x = self.relu(x)

        for layer in self.params[1:-1]:
            weights, scales, biases = layer

            weights = weights.reshape(-1, self.hidden_dim, self.hidden_dim)
            scales = scales.unsqueeze(1)
            biases = biases.unsqueeze(1)

            x = torch.bmm(x, weights) * scales + biases
            # x = self.dropout(x)
            x = self.relu(x)

        weights, scales, biases = self.params[-1]

        weights = weights.reshape(-1, self.hidden_dim, 1)
        scales = scales.unsqueeze(1)
        biases = biases.unsqueeze(1)

        x = torch.bmm(x, weights) * scales + biases

        return x
