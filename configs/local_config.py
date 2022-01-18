import subprocess
from dataclasses import dataclass
import torch

from configs import server_config

device = 'cuda'


@dataclass
class DataConfig(server_config.DataConfig):
    val_samples = 100


@dataclass
class ModelConfig(server_config.ModelConfig):
    embed_dim = 256
    encoder_depth = 2
    num_heads = 4
    attn_drop_rate = 0.2
    drop_rate = 0.2


def git_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


@dataclass
class TrainConfig(server_config.TrainConfig):
    wd = 0.0005
    mb_size = 8
    n_epoch = 20
    clip_value = 5  # 0.5?
    num_workers = 4  # TODO PUT 4
    optimizer = torch.optim.AdamW


@dataclass
class EvalConfig(server_config.EvalConfig):
    grid_res_step = 0.04
    mb_size = 32
    wandb = False
