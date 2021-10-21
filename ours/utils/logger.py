import wandb
from configs.local_config import TrainConfig, ModelConfig, DataConfig
import copy
import torch


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# TODO save a default separate step for everything in a dict with the option of specifing one
class Logger(metaclass=Singleton):
    def __init__(self, active=True):
        self.active = active
        if active:
            wandb.login(key="f5f77cf17fad38aaa2db860576eee24bde163b7a")
            wandb.init(project='pcr', entity='coredump')
            # transform class into dict to log it to wandb

    def log_model(self, model):
        if self.active:
            wandb.watch(model, log="all", log_freq=1, log_graph=True)

    def log_config(self):
        if self.active:
            wandb.config["train"] = {k: dict(TrainConfig.__dict__)[k] for k in dict(TrainConfig.__dict__) if
                                     not k.startswith("__")}
            wandb.config["model"] = {k: dict(ModelConfig.__dict__)[k] for k in dict(ModelConfig.__dict__) if
                                     not k.startswith("__")}
            wandb.config["data"] = {k: dict(DataConfig.__dict__)[k] for k in dict(DataConfig.__dict__) if
                                not k.startswith("__")}

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

    def log_config_file(self):
        if self.active:
            pass
            # wandb.log({"ours/configs/local_config.py"})