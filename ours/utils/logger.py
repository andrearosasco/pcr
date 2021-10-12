import wandb
from configs.cfg1 import TrainConfig, ModelConfig, DataConfig
import copy
import torch


class Logger:
    def __init__(self, model, active=True):
        self.active = active
        if active:
            wandb.login(key="f5f77cf17fad38aaa2db860576eee24bde163b7a")
            wandb.init(project='pcr', entity='coredump')
            # transform class into dict to log it to wandb TODO it does not work
            wandb.config["train"] = {k: dict(TrainConfig.__dict__)[k] for k in dict(TrainConfig.__dict__) if
                                     not k.startswith("__")}
            wandb.config["model"] = {k: dict(ModelConfig.__dict__)[k] for k in dict(ModelConfig.__dict__) if
                                     not k.startswith("__")}
            wandb.config["data"] = {k: dict(DataConfig.__dict__)[k] for k in dict(DataConfig.__dict__) if
                                    not k.startswith("__")}
            wandb.watch(model, log="all", log_freq=1, log_graph=True)

    def log_metrics(self, losses, accuracies, out_sig=None, step=None, mode="train"):
        """
        :param mode: one in "train" and "valid"
        :param step: step inside the batch
        :param losses: list ( float )
        :param accuracies: list ( float )
        :param out_sig: FloatTensor on CUDA
        """
        if self.active:
            loss = sum(losses) / len(losses)
            acc = sum(accuracies) / len(accuracies)
            if out_sig is not None:
                out_sig = out_sig.detach().cpu()

            wandb.log({mode + "/accuracy": acc})
            wandb.log({mode + "/loss": loss})
            if step is not None:
                wandb.log({mode + "/step": step})
            if out_sig is not None:
                wandb.log({mode + "/out_sig": out_sig, 'step': step})

    def log_pcs(self, complete, partial, impl_input, impl_pred):
        """
        :param complete: FloatTensor on CUDA with shape (N, 3)
        :param partial: FloatTensor on CUDA with shape (N, 3)
        :param impl_input: FloatTensor on CUDA with shape (N, 4) (also class)
        :param impl_pred: FloatTensor on CUDA with shape (N, 3)
        """
        if self.active:
            complete = complete.detach().cpu().numpy()
            partial = partial.detach().cpu().numpy()
            impl_input = impl_input.detach().cpu().numpy()
            impl_pred = impl_pred.detach().cpu().numpy()

            pcs = {}
            for pc, name in zip([complete, partial, impl_input, impl_pred],
                                ["original", "partial", "impl_input", "impl_pred"]):
                pcs[name] = wandb.Object3D({"type": "lidar/beta", "points": pc})
            wandb.log(pcs)

    def log_recon(self, recon):
        if self.active:
            recon = recon.detach().cpu().numpy()
            recon = wandb.Object3D({"type": "lidar/beta", "points": recon})
            wandb.log({"reconstruction": recon})
