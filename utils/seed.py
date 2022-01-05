import os
import random
import numpy as np
import torch
from configs import TrainConfig


def seed_worker(worker_id):
    worker_seed = TrainConfig.seed % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def enable_reproducibility(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


