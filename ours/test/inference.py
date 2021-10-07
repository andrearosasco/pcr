from math import sqrt
import torch
from utils.misc import pc_grid_reconstruction
import open3d as o3d


def sphere_model(points):  # N 3 -> N 1 TODO find a better way
    dist = torch.sqrt(torch.square(points[:, 0]) + torch.square(points[:, 1]) + torch.square(points[:, 2]))
    return (dist < 0.4).unsqueeze(-1)


def cube_model(points):  # N 3 -> N 1 TODO find a better way
    dist = 0.3 < points[:, 0]  # < 0.7 and 0.3 < points[:, 1] < 0.7 and 0.3 < points[:, 2] < 0.7
    return dist.unsqueeze(-1)


def combo_model(batches):  # B N 3
    results = []
    for i in range(len(batches)):
        results.append(sphere_model(batches[0]))
    return torch.stack(results)


if __name__ == "__main__":
    import time
    import tqdm
    model = combo_model
    times = []

    # pc_grid_reconstruction(model, bs=1024)
    #
    # for i in tqdm.tqdm(range(3, 1024, 10), position=0, leave=True):
    #     start = time.time()
    #     print(i)
    #     pc_grid_reconstruction(model, bs=i)
    #     end = time.time()
    #     times.append(end - start)
    #
    # import matplotlib.pyplot as plt
    # plt.plot(times)
    # plt.show()

    res = pc_grid_reconstruction(model, bs=3)

    for r in res:

        # Convert binary results into colors
        c = []
        for v in r:
            if v[-1] == 0.:
                c.append(torch.FloatTensor([1., 0., 0.]))
            else:
                c.append(torch.FloatTensor([0., 1., 0.]))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(r[:, :-1].cpu())
        pcd.colors = o3d.utility.Vector3dVector(torch.stack(c).cpu())
        o3d.visualization.draw_geometries([pcd])
