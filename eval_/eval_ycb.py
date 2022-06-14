import math
import os
import sys
from pathlib import Path

import wandb

from configs import Config

from utils.lightning import SplitProgressBar
from utils.reproducibility import make_reproducible

os.environ['CUDA_VISIBLE_DEVICES'] = Config.General.visible_dev
from pytorch_lightning.callbacks import GPUStatsMonitor, ProgressBar, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.loggers import WandbLogger


from model import PCRNetwork as Model
import pytorch_lightning as pl


def main():
    make_reproducible(Config.General.seed)
    model = Model(Config.Model)
    # model.to('cuda')
    # print_memory()
    id = '2pw4byh5'
    project = 'pcr'
    ckpt = f'model-{id}:v21'

    # ckpt.replace(':', '-')
    file = Path('artifacts') / ckpt.replace(':', '-') / 'model.ckpt'  # on windows: .replace(':', '-')

    run = wandb.init(id=id)
    name = run.name
    if not file.exists():
        run.use_artifact(f'rosasco/{project}/{ckpt}', type='model').download(f'artifacts/{ckpt}/')
    wandb.finish(exit_code=0)

    wandb_logger = WandbLogger(project='eval_ycb', log_model='all')
    wandb_logger.watch(model)

    # checkpoint = torch.load('./checkpoint/20-10-21_0732.ptc')
    # aux = {}
    # aux['state_dict'] = checkpoint
    # torch.save(aux, './checkpoint/best.ptc')
    model = Model.load_from_checkpoint(str(file), config=Config.Model, strict=False)

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