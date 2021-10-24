import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace

import torch

device = 'cuda'

@dataclass
class DataConfig:
    dataset_path = "../data/ShapeNetCore.v2"
    partial_points = 2048
    multiplier_complete_sampling = 5
    noise_rate = 0.02  # amount of noise added to the point sampled on the mesh
    percentage_sampled = 0.1  # number of uniformly sampled points
    mode = 'easy'  # values: train, valid, test
    n_classes = 55 # TODO re-label classes for easy dataset
    implicit_input_dimension = 8192


@dataclass
class ModelConfig:
    PC_SIZE = 2048
    knn_layer = 1
    device = device
    # Transformer
    n_channels = 3
    embed_dim = 384
    encoder_depth = 6
    mlp_ratio = 2.
    qkv_bias = False
    num_heads = 6
    attn_drop_rate = 0. # TODO non stiamo usando il dropout da nessuna parte?
    drop_rate = 0.
    qk_scale = None
    out_size = 1024
    # Implicit Function
    hidden_dim = 32
    depth = 2
    # Others
    use_object_id = False
    use_deep_weights_generator = False
    n_classes = 55  # TODO re-label classes for easy dataset
    assert divmod(embed_dim, num_heads)[1] == 0


def git_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


@dataclass
class TrainConfig:
    device = device
    visible_dev = '1'
    lr = 1e-4
    mb_size = 64 # TODO perché serve così basso adesso?
    n_epoch = 20
    clip_value = 5 # 0.5?
    log_metrics_every = 100
    log_pcs_every = 10000
    seed = 1   # 1234 5678 does not converge int(datetime.now().timestamp())
    num_workers = 20
    git = git_hash()
    optimizer = torch.optim.Adam
    loss = torch.nn.BCEWithLogitsLoss
    loss_reduction = "mean"  # "none"
    load_ckpt = None
    save_ckpt = f"{datetime.now().strftime('%d-%m-%y_%H-%M')}"
    overfit_mode = False
    # overfit_sample = "../data/ShapeNetCore.v2/02747177/1ce689a5c781af1bcf01bc59d215f0/models/model_normalized.obj"
    # overfit_sample = "../pcr/data/ShapeNetCore.v2/02691156/1a9b552befd6306cc8f2d5fe7449af61/models/model_normalized.obj"
    grid_res_step = 0.02


# NewConfig = SimpleNamespace(
#     visible_dev='0',
#     implicit=SimpleNamespace(
#         hidden_dim=32,
#         depth=2
#     )
# )
#
# print(NewConfig.implicit.hidden_dim)