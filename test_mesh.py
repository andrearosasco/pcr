from pathlib import Path

import numpy as np
import torch
from dgl.geometry import farthest_point_sampler

from configs.debug_config import Config
import os

from datasets.GraspingDataset import GraspingDataset
from utils.misc import voxelize_pc

os.environ['CUDA_VISIBLE_DEVICES'] = Config.General.visible_dev
from model.PCRNetwork2 import PCRNetwork as Model
import wandb
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
import open3d as o3d


def main():
    id = '1m5301hl'
    ckpt = f'model-{id}:v53'
    project = 'pcr-grasping'

    ckpt_path = f'artifacts/{ckpt}/model.ckpt' if os.name != 'nt' else \
        f'artifacts/{ckpt}/model.ckpt'.replace(':', '-')

    if not Path(ckpt_path).exists():
        run = wandb.init(id=id, settings=wandb.Settings(start_method="spawn"))
        run.use_artifact(f'rosasco/{project}/{ckpt}', type='model').download(f'artifacts/{ckpt}/')
        wandb.finish(exit_code=0)

    model = Model.load_from_checkpoint(ckpt_path, config=Config.Model)
    model.cuda()
    model.eval()

    ds = GraspingDataset(Config.Data.dataset_path, 'data/MCD/build_datasets/train_test_dataset.json', subset='holdout_models_holdout_views')

    for data in ds:
        partial, ground_truth = data
        partial = torch.tensor(partial, device=Config.General.device, dtype=torch.float32).unsqueeze(0)
        ground_truth = torch.tensor(ground_truth, device=Config.General.device, dtype=torch.float32).unsqueeze(0)

        reconstruction, probailities = model(partial, num_points=8192*2)

        aux, _ = model(partial, num_points=200_000)
        point_idx = farthest_point_sampler(aux, 8192*2)
        sampled_rec = aux[torch.arange(point_idx.shape[0]).unsqueeze(-1), point_idx]

        fp_rec = PointCloud(points=Vector3dVector(sampled_rec[0].cpu().numpy())).paint_uniform_color([1, 1, 0])
        fp_rec.estimate_normals()
        rec = PointCloud(points=Vector3dVector(reconstruction[0].cpu().numpy())).paint_uniform_color([1, 0, 0])
        rec.estimate_normals()
        #
        # import mcubes
        # grid1 = voxelize_pc(reconstruction, 0.025)
        # grid2 = voxelize_pc(ground_truth, 0.025)
        #
        # vertices, triangles = mcubes.marching_cubes(grid1[0].cpu().numpy(), 0.5)
        # mesh = o3d.geometry.TriangleMesh(triangles=Vector3iVector(triangles), vertices=Vector3dVector(vertices))
        #
        # jaccard = torch.sum(grid1 * grid2) / torch.sum((grid1 + grid2) != 0)
        # # o3d.visualization.draw_geometries([mesh])
        #
        # print(jaccard)

        o3d.visualization.draw(
            [fp_rec, rec,
             PointCloud(points=Vector3dVector(partial[0].cpu().numpy())).paint_uniform_color([0, 1, 0]),
             PointCloud(points=Vector3dVector(ground_truth[0].cpu().numpy())).paint_uniform_color([0, 0, 1])])

        print()

if __name__ == '__main__':
    main()