import math

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger

from configs import Config
from model.PCRNetwork import PCRNetwork as Model
from utils.lightning import SplitProgressBar

if __name__ == '__main__':
    # id = '2pw4byh5'
    ckpt = 'model-2pw4byh5:v20'

    model = Model(Config.Model)

    config = Config.to_dict()

    # run = wandb.init(project='pcr', id=id, resume='must')
    run = wandb.init(project='pcr')
    artifact = run.use_artifact(f'rosasco/pcr/{ckpt}', type='model')
    # .replace(':', '-')
    artifact_dir = artifact.download(f"artifacts/{ckpt}/")

    wandb_logger = WandbLogger(project='pcr', log_model='all', config=config, resume="allow",
                               reinit=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        mode='max',
        auto_insert_metric_name=False)

    trainer = Trainer(max_epochs=Config.Train.n_epoch,
                      precision=32,
                      gpus=1,
                      log_every_n_steps=Config.Train.log_metrics_every,
                      logger=[wandb_logger],
                      gradient_clip_val=Config.Train.clip_value,
                      gradient_clip_algorithm='value',
                      callbacks=[DeviceStatsMonitor(),
                                 SplitProgressBar(),
                                 checkpoint_callback],
                      )

    # trainer.validate(model, ckpt_path=f"artifacts/{ckpt.replace(':', '-')}/model.ckpt")
    trainer.fit(model, ckpt_path=f'artifacts/{ckpt}/model.ckpt')
