import math
import os
import sys

from configs import DataConfig, ModelConfig, TrainConfig, server_config, EvalConfig
from utils.lightning import SplitProgressBar
from utils.reproducibility import make_reproducible

os.environ['CUDA_VISIBLE_DEVICES'] = TrainConfig.visible_dev
from pytorch_lightning.callbacks import GPUStatsMonitor, ProgressBar, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.loggers import WandbLogger

import random
import numpy as np
from model import PCRNetwork as Model
import torch

import pytorch_lightning as pl


if __name__ == '__main__':
    make_reproducible(TrainConfig.seed)
    model = Model(ModelConfig)
    # model.to('cuda')
    # print_memory()

    wandb_logger = WandbLogger(project='pcr', log_model='all', entity='coredump')
    wandb_logger.watch(model)

    # checkpoint = torch.load('./checkpoint/20-10-21_0732.ptc')
    # aux = {}
    # aux['state_dict'] = checkpoint
    # torch.save(aux, './checkpoint/best.ptc')
    model = Model.load_from_checkpoint('./checkpoint/best.ptc', config=server_config.ModelConfig, )

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        auto_insert_metric_name=False)

    trainer = pl.Trainer(max_epochs=TrainConfig.n_epoch,
                         precision=32,
                         gpus=1,
                         log_every_n_steps=EvalConfig.log_metrics_every,
                         logger=[wandb_logger],
                         gradient_clip_val=TrainConfig.clip_value,
                         gradient_clip_algorithm='value',
                         callbacks=[GPUStatsMonitor(),
                                    SplitProgressBar(),
                                    checkpoint_callback],
                         )

    # trainer.fit(model)
    trainer.validate(model)
