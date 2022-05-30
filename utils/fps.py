from torch import cdist
import torch


def fp_sampling(points, num):
    batch_size = points.shape[0]
    D = cdist(points, points)
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    res = torch.zeros((batch_size, num), dtype=torch.int32, device=points.device)
    ds = D[:, 0, :]
    for i in range(1, num):
        idx = torch.argmax(ds, dim=1)
        res[:, i] = idx
        ds = torch.minimum(ds, D[torch.arange(batch_size), idx, :])

    return res