import math
import os
import sys

from utils.configuration import BaseConfig as Config
from utils.lightning import SplitProgressBar
from utils.reproducibility import make_reproducible

os.environ['CUDA_VISIBLE_DEVICES'] = Config.General.visible_dev
from pytorch_lightning.callbacks import GPUStatsMonitor, ProgressBar, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.loggers import WandbLogger

import random
import numpy as np
from model import PCRNetwork as Model
import torch

import pytorch_lightning as pl


if __name__ == '__main__':
    make_reproducible(Config.General.seed)
    model = Model(Config.Model)
    # model.to('cuda')
    # print_memory()

    wandb_logger = WandbLogger(project='pcr', log_model='all')
    wandb_logger.watch(model)

    # checkpoint = torch.load('./checkpoint/20-10-21_0732.ptc')
    # aux = {}
    # aux['state_dict'] = checkpoint
    # torch.save(aux, './checkpoint/best.ptc')
    model = Model.load_from_checkpoint('./checkpoint/best.ptc', config=Config.Model, )

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        auto_insert_metric_name=False)

    trainer = pl.Trainer(max_epochs=Config.Train.n_epoch,
                         precision=32,
                         gpus=1,
                         log_every_n_steps=Config.Eval.log_metrics_every,
                         logger=[wandb_logger],
                         gradient_clip_val=Config.Train.clip_value,
                         gradient_clip_algorithm='value',
                         callbacks=[GPUStatsMonitor(),
                                    SplitProgressBar(),
                                    checkpoint_callback],
                         )

    # trainer.fit(model)
    trainer.validate(model)
