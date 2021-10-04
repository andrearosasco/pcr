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

    def log_metrics(self, losses, accuracies, out, out_sig, target):
        """
        :param losses: list ( float )
        :param accuracies: list ( float )
        :param out: FloatTensor on CUDA
        :param out_sig: FloatTensor on CUDA
        :param target: FloatTensor on CUDA
        :param z: FloatTensor on CUDA
        :param impl_params: list ( list ( FloatTensor on CUDA ) )
        """
        if self.active:
            loss = sum(losses) / len(losses)
            acc = sum(accuracies) / len(accuracies)
            out = out.detach().cpu()
            out_sig = out_sig.detach().cpu()
            pred = copy.deepcopy(out_sig).apply_(lambda v: 1 if v > 0.5 else 0)
            target = target.detach().cpu()
            # z = z.detach().cpu()

            wandb.log({"accuracy": acc})
            wandb.log({"loss": loss})
            wandb.log({"out": out})
            wandb.log({"out_sig": out_sig})
            wandb.log({"pred": pred})
            wandb.log({"target": target})
            # wandb.log({"z": z})

            weights = []
            scales = []
            biases = []
            # for param in impl_params:
            #     weights.append(param[0].view(-1))
            #     scales.append(param[1].view(-1))
            #     biases.append(param[2].view(-1))
            # wandb.log({"w of implicit function": torch.cat(weights)})
            # wandb.log({"scales of implicit function": torch.cat(scales)})
            # wandb.log({"b of implicit function": torch.cat(biases)})

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
