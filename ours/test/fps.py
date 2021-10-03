import copy

from torch import cdist
import torch


def fp_sampling(points, num):
    batch_size = points.shape[0]
    D = cdist(points[:, 0].unsqueeze(1), points).squeeze(1)
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = torch.zeros((batch_size, num), dtype=torch.int32, device=points.device)
    ds = D
    for i in range(1, num):
        idx = torch.argmax(ds, dim=1)
        perm[:, i] = idx
        ds = torch.minimum(ds, cdist(points[torch.arange(batch_size), idx].unsqueeze(1), points).squeeze())

    return perm
#
# def fp_sampling(points, num):
#     batch_size = points.shape[0]
#     D = cdist(points, points)
#     # By default, takes the first point in the list to be the
#     # first point in the permutation, but could be random
#     res = torch.zeros((batch_size, num), dtype=torch.int32, device=points.device)
#     ds = D[:, 0, :]
#     for i in range(1, num):
#         idx = torch.argmax(ds, dim=1)
#         res[:, i] = idx
#         ds = torch.minimum(ds, D[torch.arange(batch_size), idx, :])
#
#     return res


ps = torch.rand((1, 5371, 3))

idxs = fp_sampling(ps, 2048)
new_combined_x = ps[torch.arange(idxs.shape[0]).unsqueeze(-1), idxs.long(), :]

