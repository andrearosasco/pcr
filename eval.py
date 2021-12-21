import math
import os
from configs import ModelConfig, TrainConfig, server_config
os.environ['CUDA_VISIBLE_DEVICES'] = TrainConfig.visible_dev  # TODO TEST TO MOVE AFTER LIGHTNING
from pytorch_lightning.callbacks import GPUStatsMonitor, ProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from models.HyperNetwork import HyperNetwork
import pytorch_lightning as pl


# ================================================================


if __name__ == '__main__':
    # print_memory()
    model = HyperNetwork(ModelConfig)
    # model.to('cuda')
    # print_memory()

    wandb_logger = WandbLogger(project='pcr', log_model='all', entity='coredump')
    wandb_logger.watch(model)

    # checkpoint = torch.load('./checkpoint/20-10-21_0732.ptc')
    # aux = {}
    # aux['state_dict'] = checkpoint
    # torch.save(aux, './checkpoint/best.ptc')
    model = HyperNetwork.load_from_checkpoint('./checkpoint/best.ptc', config=server_config.ModelConfig, )

    checkpoint_callback = ModelCheckpoint(
        monitor='valid/f1',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-f1{valid/f1:.2f}',
        auto_insert_metric_name=False)


    class LitProgressBar(ProgressBar):

        def on_train_epoch_start(self, trainer, pl_module):
            super().on_train_epoch_start(trainer, pl_module)
            total_train_batches = self.total_train_batches

            total_batches = total_train_batches
            if total_batches is None or math.isinf(total_batches) or math.isnan(total_batches):
                total_batches = None
            if not self.main_progress_bar.disable:
                self.main_progress_bar.reset(total=total_batches)
            self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")


    bar = LitProgressBar()

    trainer = pl.Trainer(max_epochs=TrainConfig.n_epoch,
                         precision=32,
                         gpus=1,
                         log_every_n_steps=TrainConfig.log_metrics_every,
                         logger=[wandb_logger],
                         gradient_clip_val=TrainConfig.clip_value,
                         gradient_clip_algorithm='value',
                         callbacks=[GPUStatsMonitor(),
                                    bar,
                                    checkpoint_callback],
                         )

    # trainer.fit(model)
    trainer.validate(model)
