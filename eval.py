import os
from configs import Config
os.environ['CUDA_VISIBLE_DEVICES'] = Config.General.visible_dev

from pathlib import Path
import wandb
from utils.lightning import SplitProgressBar
from utils.reproducibility import make_reproducible
from pytorch_lightning.callbacks import GPUStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from model import PCRNetwork as Model

import pytorch_lightning as pl


def main():
    make_reproducible(Config.General.seed)

    id = '3ny0ctj1' # '2u76m6gp'
    ckpt = f'model-{id}:v103' # '400'
    project = 'pcr'

    file = Path('artifacts') / ckpt.replace(':', '-') / 'model.ckpt'  # on windows: .replace(':', '-')

    run = wandb.init(id=id)
    name = run.name
    run.use_artifact(f'rosasco/{project}/{ckpt}', type='model').download(f'artifacts/{ckpt}/')
    wandb.finish(exit_code=0)

    model = Model.load_from_checkpoint(str(file), config=Config.Model)

    logger = []
    # wandb_logger = WandbLogger(name='eval_shapenet', project='pcr', log_model='all')
    # wandb_logger.watch(model)
    # logger.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        auto_insert_metric_name=False)

    trainer = pl.Trainer(max_epochs=Config.Train.n_epoch,
                         precision=32,
                         gpus=1,
                         log_every_n_steps=Config.Eval.log_metrics_every,
                         logger=logger,
                         gradient_clip_val=Config.Train.clip_value,
                         gradient_clip_algorithm='value',
                         callbacks=[GPUStatsMonitor(),
                                    SplitProgressBar(),
                                    checkpoint_callback],
                         )

    trainer.validate(model)
