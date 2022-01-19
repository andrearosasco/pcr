import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor
from pytorch_lightning.loggers import WandbLogger
from configs import TrainConfig, ModelConfig, DataConfig, EvalConfig
from model import PCRNetwork as Model
from utils.lightning import SplitProgressBar

if __name__ == '__main__':
    id = '29o44g6w'
    ckpt = 'model-29o44g6w:v29'

    model = Model(ModelConfig)

    config = {'train': {k: dict(TrainConfig.__dict__)[k] for k in dict(TrainConfig.__dict__) if
                        not k.startswith("__")},
              'model': {k: dict(ModelConfig.__dict__)[k] for k in dict(ModelConfig.__dict__) if
                        not k.startswith("__")},
              'data': {k: dict(DataConfig.__dict__)[k] for k in dict(DataConfig.__dict__) if
                       not k.startswith("__")}}

    run = wandb.init(project='train_box', id=id, resume='must')

    artifact = run.use_artifact(f'rosasco/train_box/{ckpt}', type='model')
    artifact_dir = artifact.download(f'artifacts/{ckpt}/')
    wandb_logger = WandbLogger(project='train_box', log_model='all', config=config)
    wandb.watch(model, log='all', log_freq=EvalConfig.log_metrics_every)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        mode='max',
        auto_insert_metric_name=False)

    trainer = Trainer(max_epochs=TrainConfig.n_epoch,
                      precision=32,
                      gpus=1,
                      log_every_n_steps=EvalConfig.log_metrics_every,
                      check_val_every_n_epoch=EvalConfig.val_every,
                      logger=[wandb_logger],
                      gradient_clip_val=TrainConfig.clip_value,
                      gradient_clip_algorithm='value',
                      callbacks=[
                                 SplitProgressBar(),
                                 checkpoint_callback],
                      )

    trainer.fit(model, ckpt_path=f'artifacts/{ckpt}/model.ckpt')
