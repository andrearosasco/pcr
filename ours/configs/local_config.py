import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import torch
import platform


# PARAMETERS ###########################################################################################################

device = "cuda"
local = "Windows" in platform.platform()


@dataclass
class DataConfig:
    dataset_path = "../data/ShapeNetCore.v2"
    partial_points = 2048
    multiplier_complete_sampling = 3
    noise_rate = 0.02
    percentage_sampled = 0.1
    mode = 'train'  # train, valid, test
    n_classes = 55


@dataclass
class ModelConfig:
    PC_SIZE = 2048
    knn_layer = 1
    device = device
    # Transformer
    n_channels = 3
    embed_dim = 384
    encoder_depth = 2 if local else 6
    mlp_ratio = 2.
    qkv_bias = False
    num_heads = 6
    attn_drop_rate = 0.
    drop_rate = 0.
    qk_scale = None
    out_size = 1024
    # Implicit Function
    hidden_dim = 32
    # Others
    use_object_id = True
    use_deep_weights_generator = True
    n_classes = 55


def git_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


@dataclass
class TrainConfig:
    difficulty = "easy"
    device = device
    visible_dev = '0'
    mb_size = 8
    n_epoch = 20
    clip_value = 5
    log_metrics_every = 1 if local else 100
    log_pcs_every = 10000
    seed = 1   # 1234 5678 does not converge int(datetime.now().timestamp())
    num_workers = 4 if local else 20
    git = git_hash()
    optimizer = torch.optim.Adam
    loss = torch.nn.BCEWithLogitsLoss

    load_ckpt = None
    save_ckpt = datetime.now().strftime('%d-%m-%y_%H:%M')
