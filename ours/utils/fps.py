from torch import cdist
import torch


def fp_sampling(points, num):
    if points.shape[0] == 8:
        pass

    D = cdist(points, points)
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = torch.zeros(num, dtype=torch.int32, device=points.device)
    ds = D[0, :]
    for i in range(1, num):
        idx = torch.argmax(ds)
        perm[i] = idx

        ds = torch.minimum(ds, D[idx, :])
    return perm.unsqueeze(0)
