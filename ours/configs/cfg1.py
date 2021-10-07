from pathlib import Path
from dataclasses import dataclass


# PARAMETERS ###########################################################################################################

device = "cuda"


@dataclass
class DataConfig:
    DATA_PATH = Path("../data") / "ShapeNet55-34" / "ShapeNet-55"
    NAME = "ShapeNet"
    N_POINTS = 8192
    subset = "train"
    PC_PATH = Path("../data") / "ShapeNet55-34" / "shapenet_pc"
    voxel_size = 0.1
    noise_rate = 0.1
    percentage_sampled = 0.1


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


@dataclass
class TrainConfig:
    difficulty = "easy"
    device = device
    visible_dev = '0'
    mb_size = 8
    n_epoch = 20
    log_metrics_every = 1
    log_pcs_every = 10000
    seed = 5678   # 1234 does not converge
    num_workers = 8
