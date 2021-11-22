import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace

import torch

device = 'cuda'
# TODO:
# Trying lower learning rate
# Try using warm-up
# Try different sampling
# Try regularization
# Trying reducing one hot dimension for easy setup (55 -> 10)
# Trying putting the one hot through a linear layer and make it 1024 before concat
# Try a different initialization
# Try taking the mean of the global features

@dataclass
class DataConfig:
    dataset_path = "../data/ShapeNetCore.v2"
    partial_points = 2024 # TODO Possible OOM
    multiplier_complete_sampling = 50
    noise_rate = 0.02  # amount of noise added to the point sampled on the mesh
    percentage_sampled = 0.1  # number of uniformly sampled points
    mode = 'easy'  # values: train, valid, test
    n_classes = 55 # TODO re-label classes for easy dataset
    implicit_input_dimension = 8192


@dataclass
class ModelConfig:
    PC_SIZE = 2048
    knn_layer = 1  # TODO Prima era 1
    device = device
    # Transformer
    n_channels = 3
    embed_dim = 384
    encoder_depth = 6
    mlp_ratio = 2.
    qkv_bias = False
    num_heads = 6
    attn_drop_rate = 0.  # TODO non stiamo usando il dropout da nessuna parte? Prima era a 0
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
    visible_dev = '0'
    lr = 1e-4
    wd = 0.0
    mb_size = 64
    test_mb_size = 32
    n_epoch = 100
    clip_value = 1 # 0.5?
    log_metrics_every = 100
    seed = 1   # 1234 5678 does not converge int(datetime.now().timestamp())
    # WARNING: Each worker load a different batches so we may end up with
    #   20 * 64 batches loaded simultaneously. Moving the batches to cuda inside the
    #   dataset can lead to OOM errors
    num_workers = 20
    git = git_hash()
    optimizer = torch.optim.Adam
    loss = torch.nn.BCEWithLogitsLoss
    loss_reduction = "norm"  # "none"
    load_ckpt = None
    save_ckpt = f"{datetime.now().strftime('%d-%m-%y_%H-%M')}"
    overfit_mode = False
    # overfit_sample = "../data/ShapeNetCore.v2/02747177/1ce689a5c781af1bcf01bc59d215f0/models/model_normalized.obj"
    # overfit_sample = "../pcr/data/ShapeNetCore.v2/02691156/1a9b552befd6306cc8f2d5fe7449af61/models/model_normalized.obj"
    grid_res_step = 0.04


# NewConfig = SimpleNamespace(
#     visible_dev='0',
#     implicit=SimpleNamespace(
#         hidden_dim=32,
#         depth=2
#     )
# )
#
# print(NewConfig.implicit.hidden_dim)
