import os
import random
import numpy
import torch


def make_reproducible(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    numpy.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_init_fn(seed):
    def init_fn():
        numpy.random.seed(seed)
        random.seed(seed)

    return init_fn()

