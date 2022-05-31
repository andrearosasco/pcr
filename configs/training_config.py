import subprocess

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
        num_workers = 30
        git = git_hash()

    class Train:
        lr = 1e-5
        wd = 0.0
        n_epoch = 100
        mb_size = 64
        clip_value = 1
        optimizer = torch.optim.Adam
        loss = torch.nn.BCEWithLogitsLoss
        loss_reduction = "mean"  # "none"

    class Eval:
        wandb = True
        log_metrics_every = 100
        val_every = 10

        grid_eval = False
        grid_res_step = 0.04

    class Data:
        dataset_path = "../data/ShapeNetCore.v2"
        partial_points = 2048  # number of points per input
        multiplier_complete_sampling = 50
        implicit_input_dimension = 8192
        dist = [0.1, 0.4, 0.5]
        noise_rate = 0.01
        tolerance = 0.0
        n_classes = 55
        noise = False

        class Eval:
            mb_size = 64

        class Train:
            mb_size = 64

    class Model:
        knn_layer = 1
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
        depth = 2
        # Others
        use_object_id = False
        use_deep_weights_generator = False
        assert divmod(embed_dim, num_heads)[1] == 0


if __name__ == '__main__':
    Config()