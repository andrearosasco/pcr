import subprocess
from pathlib import Path

import torch

from utils.configuration import BaseConfig


def git_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


class Config(BaseConfig):

    run = 'train'

    class General:
        device = 'cuda'
        visible_dev = '0'
        seed = 1
        num_workers = 0
        git = git_hash()

    class Train:
        lr = 1e-5
        wd = 0.0
        n_epoch = 100
        clip_value = 1
        optimizer = torch.optim.Adam
        loss = torch.nn.BCEWithLogitsLoss
        loss_reduction = "mean"  # "none"

        mb_size = 8

    class Eval:
        wandb = False
        log_metrics_every = 100
        val_every = 1

        grid_eval = False
        grid_res_step = 0.04

        mb_size = 8

    class Data:
        dataset_path = "./data/ShapeNetCore.v2"
        mode = 'hard'
        partial_points = 2048  # number of points per input
        multiplier_complete_sampling = 50
        implicit_input_dimension = 8192
        dist = [0.1, 0.4, 0.5]
        noise_rate = 0.01
        tolerance = 0.0
        n_classes = 55
        offset = True

        class Eval:
            mb_size = 8

        class Train:
            mb_size = 8

    class Model:
        knn_layer = 1
        # Transformer
        n_channels = 3
        embed_dim = 256
        encoder_depth = 2
        mlp_ratio = 2.
        qkv_bias = False
        num_heads = 4
        attn_drop_rate = 0.2
        drop_rate = 0.2
        qk_scale = None
        out_size = 1024
        # Implicit Function
        hidden_dim = 32
        depth = 0
        # Others
        use_object_id = False
        use_deep_weights_generator = False
        assert divmod(embed_dim, num_heads)[1] == 0


if __name__ == '__main__':
    import train
    with Path('configs/__init__.py').open('w+') as f:
        f.writelines(['from .debug_config import Config'])
    train.main()
