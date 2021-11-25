import math
import os
import sys

from torch import nn

from utils.lightning import SplitProgressBar

try:
    from open3d.cuda.pybind.utility import Vector3dVector
    from open3d.cuda.pybind.visualization import draw_geometries
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    from open3d.cpu.pybind.utility import Vector3dVector
    from open3d.cpu.pybind.visualization import draw_geometries
    from open3d.cpu.pybind.geometry import PointCloud

from configs import DataConfig, ModelConfig, TrainConfig

os.environ['CUDA_VISIBLE_DEVICES'] = TrainConfig.visible_dev
from pytorch_lightning.callbacks import GPUStatsMonitor, ProgressBar, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, Precision, Recall, F1, AverageMeter

from utils.logger import Logger
import random
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from datasets.ShapeNetPOVRemoval import ShapeNet
from models.HyperNetwork import BackBone, ImplicitFunction
import torch
from tqdm import tqdm
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






if __name__ == '__main__':
    model = HyperNetwork(ModelConfig)

    config = {'train': {k: dict(TrainConfig.__dict__)[k] for k in dict(TrainConfig.__dict__) if
                        not k.startswith("__")},
              'model': {k: dict(ModelConfig.__dict__)[k] for k in dict(ModelConfig.__dict__) if
                        not k.startswith("__")},
              'data': {k: dict(DataConfig.__dict__)[k] for k in dict(DataConfig.__dict__) if
                       not k.startswith("__")}}

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