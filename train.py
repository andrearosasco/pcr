from utils.reproducibility import make_reproducible
from utils.lightning import SplitProgressBar

# try:
#     from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
#     from open3d.cuda.pybind.visualization import draw_geometries
#     from open3d.cuda.pybind.geometry import PointCloud
# except ImportError:

    # from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    # from open3d.cpu.pybind.visualization import draw_geometries
    # from open3d.cpu.pybind.geometry import PointCloud

from configs import Config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = Config.General.visible_dev
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from model.PCRNetwork import PCRNetwork as Model
import pytorch_lightning as pl
import wandb
import torch

def main():
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # make_reproducible(TrainConfig.seed)

    model = Model(Config.Model)

    loggers = []
    resume = False
    ckpt = 'model-2onk4j45:v3'
    ckpt_path = None
    project = 'pcr'
    if Config.Eval.wandb:
        if resume:
            wandb_logger = WandbLogger(project=project, id='2onk4j45', log_model='all', resume='must', config=Config.to_dict())
            run = wandb_logger.experiment

            artifact = run.use_artifact(f'rosasco/pcr/{ckpt}', type='model')
            artifact_dir = artifact.download(f"artifacts/{ckpt}/") # .replace(':', '-')
            ckpt_path = f'artifacts/{ckpt}/model.ckpt'
        else:
            wandb_logger = WandbLogger(project=project, log_model='all', config=Config.to_dict())

        loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        mode='max',
        save_last=True,
        auto_insert_metric_name=False)

    trainer = pl.Trainer(max_epochs=Config.Train.n_epoch,
                         precision=32,
                         gpus=1,
                         log_every_n_steps=Config.Eval.log_metrics_every,
                         check_val_every_n_epoch=Config.Eval.val_every,
                         logger=loggers,
                         gradient_clip_val=Config.Train.clip_value,
                         gradient_clip_algorithm='value',
                         num_sanity_val_steps=2,
                         callbacks=[
                                    SplitProgressBar(),
                                    checkpoint_callback],
                         )

    trainer.fit(model, ckpt_path=ckpt_path)