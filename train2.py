from pathlib import Path

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
from model.PCRNetwork2 import PCRNetwork as Model
import pytorch_lightning as pl
import wandb

def main():
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # make_reproducible(TrainConfig.seed)
    # wandb.init(settings=wandb.Settings(start_method="fork"))
    model = Model(Config.Model)

    id = '3gvgzgmq'
    ckpt = f'model-{id}:v28'
    project = 'pcr-grasping'

    ckpt_path = None
    loggers = []
    resume = True
    use_checkpoint = True

    if use_checkpoint:
        ckpt_path = f'artifacts/{ckpt}/model.ckpt' # .replace(':', '-')

        if not Path(ckpt_path).exists():
            run = wandb.init(id=id, settings=wandb.Settings(start_method="spawn"))
            run.use_artifact(f'rosasco/{project}/{ckpt}', type='model').download(f'artifacts/{ckpt}/')
            wandb.finish(exit_code=0)

        model = Model.load_from_checkpoint(ckpt_path, config=Config.Model)  #

    if Config.Eval.wandb:
        if resume:
            wandb.init(project=project, id=id, resume='must', config=Config.to_dict(), reinit=True, settings=wandb.Settings(start_method='thread'))
            wandb_logger = WandbLogger(log_model='all')

        else:
            wandb.init(project=project, reinit=True, config=Config.to_dict(),
                       settings=wandb.Settings(start_method="thread"))
            wandb_logger = WandbLogger(log_model='all')

        wandb_logger.watch(model, log='all', log_freq=Config.Eval.log_metrics_every)
        loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_views/chamfer',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-cd{val_views/chamfer:.2f}',
        mode='min',
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
                         num_sanity_val_steps=4,
                         callbacks=[
                                    SplitProgressBar(),
                                    checkpoint_callback],
                         )

    trainer.fit(model, ckpt_path=ckpt_path)  # .replace(':', '-')
