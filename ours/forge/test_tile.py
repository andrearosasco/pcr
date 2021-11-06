import numpy as np
import torch


def tile(a, dim, n_tile):
    init_dim = a.shape[dim]
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

t = torch.tensor([[1, 2, 3], [4, 4, 4]])
print(t.shape)
print(t)
print(t.shape[0])
print(t.size(0))
t = tile(t, 0, 3)
print(t.shape)
print(t)