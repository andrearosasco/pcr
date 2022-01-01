import os
from models.HyperNetwork import HyperNetwork
from utils.lightning import SplitProgressBar

try:
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cuda.pybind.visualization import draw_geometries
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cpu.pybind.visualization import draw_geometries
    from open3d.cpu.pybind.geometry import PointCloud

from configs import DataConfig, ModelConfig, TrainConfig, EvalConfig

os.environ['CUDA_VISIBLE_DEVICES'] = TrainConfig.visible_dev  # TODO TEST TO MOVE AFTER LIGHTNING
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import pytorch_lightning as pl

# ================================================================

def run():

    model = HyperNetwork(ModelConfig)

    config = {'train': {k: dict(TrainConfig.__dict__)[k] for k in dict(TrainConfig.__dict__) if
                        not k.startswith("__")},
              'model': {k: dict(ModelConfig.__dict__)[k] for k in dict(ModelConfig.__dict__) if
                        not k.startswith("__")},
              'data': {k: dict(DataConfig.__dict__)[k] for k in dict(DataConfig.__dict__) if
                       not k.startswith("__")}}

    wandb.login()
    wandb.init(project="train_box")
    wandb_logger = WandbLogger(project='pcr', log_model='all', config=config)
    wandb.watch(model, log='all', log_freq=EvalConfig.log_metrics_every)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        mode='max',
        auto_insert_metric_name=False)

    trainer = pl.Trainer(max_epochs=TrainConfig.n_epoch,
                         precision=32,
                         gpus=1,
                         log_every_n_steps=EvalConfig.log_metrics_every,
                         logger=[wandb_logger],
                         gradient_clip_val=TrainConfig.clip_value,
                         gradient_clip_algorithm='value',
                         num_sanity_val_steps=2,
                         callbacks=[
                                    SplitProgressBar(),
                                    checkpoint_callback],
                         )

    trainer.fit(model)