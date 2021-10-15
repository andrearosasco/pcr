import wandb
from configs.local_config import TrainConfig, ModelConfig, DataConfig
import copy
import torch


class Logger:
    def __init__(self, model, active=True):
        self.active = active
        if active:
            wandb.login(key="f5f77cf17fad38aaa2db860576eee24bde163b7a")
            wandb.init(project='pcr', entity='coredump')
            # transform class into dict to log it to wandb
            wandb.config["train"] = {k: dict(TrainConfig.__dict__)[k] for k in dict(TrainConfig.__dict__) if
                                     not k.startswith("__")}
            wandb.config["model"] = {k: dict(ModelConfig.__dict__)[k] for k in dict(ModelConfig.__dict__) if
                                     not k.startswith("__")}
            wandb.config["data"] = {k: dict(DataConfig.__dict__)[k] for k in dict(DataConfig.__dict__) if
                                    not k.startswith("__")}
            wandb.watch(model, log="all", log_freq=1, log_graph=True)

    def log_metrics(self, metrics):
        """
        :param metrics: dict of metrics to log
        """
        if self.active:
            wandb.log(metrics)

    def log_point_clouds(self, point_clouds):
        """
        :param point_clouds: dict of point clouds to log
        """
        if self.active:
            pcs = {}
            for name in point_clouds.keys():
                pcs[name] = wandb.Object3D({"type": "lidar/beta", "points": point_clouds[name]})
            wandb.log(pcs)
