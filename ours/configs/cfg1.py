import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass


# PARAMETERS ###########################################################################################################

device = "cuda"


@dataclass
class DataConfig:
    DATA_PATH = Path("data") / "ShapeNet55-34" / "ShapeNet-55"
    NAME = "ShapeNet"
    N_POINTS = 8192
    subset = "train"
    PC_PATH = Path("data") / "ShapeNet55-34" / "shapenet_pc"
    voxel_size = 0.1
    noise_rate = 0.02
    percentage_sampled = 0.1
    # OurShapeNet
    dataset_path = "../../data/ShapeNetCore.v2"
    mode = 'train' # train, valid, test


@dataclass
class ModelConfig:
    NAME = "PoinTr"
    PC_SIZE = 2048
    knn_layer = 1
    num_pred = 6144
    device = device
    # Transformer
    n_channels = 3
    embed_dim = 384
    encoder_depth = 6
    mlp_ratio = 2.
    qkv_bias = False
    num_heads = 6
    attn_drop_rate = 0.
    drop_rate = 0.
    qk_scale = None
    out_size = 1024
    # Implicit Function
    hidden_dim = 32


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
    log_metrics_every = 100
    log_pcs_every = 10000
    seed = int(datetime.now().timestamp())   # 1234 5678 does not converge
    num_workers = 4
    git = git_hash()

