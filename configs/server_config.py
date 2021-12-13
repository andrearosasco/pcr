import subprocess
from datetime import datetime
from dataclasses import dataclass
import torch

device = 'cuda'


@dataclass
class DataConfig:
    partial_points = 2024
    multiplier_complete_sampling = 50
    noise_rate = 0.02  # amount of noise added to the point sampled on the mesh
    percentage_sampled = 0.1  # number of uniformly sampled points
    implicit_input_dimension = 8192
    dist = [0.1, 0.9, 0]
    tolerance = 0.01
    train_samples = 10000
    val_samples = 100


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
    num_workers = 24
    git = ""  # git_hash()
    optimizer = torch.optim.Adam
    loss = torch.nn.BCEWithLogitsLoss
    loss_reduction = "norm"  # "none"
    load_ckpt = None
    save_ckpt = f"{datetime.now().strftime('%d-%m-%y_%H-%M')}"
    overfit_mode = False
    grid_res_step = 0.04
