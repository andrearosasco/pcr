import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor
from pytorch_lightning.loggers import WandbLogger
from configs import TrainConfig, ModelConfig, DataConfig
from models.HyperNetwork import HyperNetwork
from utils.lightning import SplitProgressBar

if __name__ == '__main__':
    id = 'a9p3pryz'
    ckpt = 'model-a9p3pryz:v39'

    model = HyperNetwork(ModelConfig)

    config = {'train': {k: dict(TrainConfig.__dict__)[k] for k in dict(TrainConfig.__dict__) if
                        not k.startswith("__")},
              'model': {k: dict(ModelConfig.__dict__)[k] for k in dict(ModelConfig.__dict__) if
                        not k.startswith("__")},
              'data': {k: dict(DataConfig.__dict__)[k] for k in dict(DataConfig.__dict__) if
                       not k.startswith("__")}}

    run = wandb.init(project='train_box', id=id, resume='must')

    artifact = run.use_artifact(f'stefanoberti/train_box/{ckpt}', type='model')
    artifact_dir = artifact.download(f'artifacts/{ckpt}/')
    wandb_logger = WandbLogger(project='train_box', log_model='all', config=config)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        mode='max',
        auto_insert_metric_name=False)

    trainer = Trainer(max_epochs=TrainConfig.n_epoch,
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

    trainer.fit(model, ckpt_path=f'artifacts/{ckpt}/model.ckpt')
