from pathlib import Path

import numpy as np
import torch
from dgl.geometry import farthest_point_sampler
from torch.utils.data import DataLoader
import tqdm
from tqdm import trange

from configs.pcn_training_config import Config
import os

from datasets.GraspingDataset import GraspingDataset
from utils.misc import voxelize_pc

os.environ['CUDA_VISIBLE_DEVICES'] = Config.General.visible_dev
from model.PCRNetwork2 import PCRNetwork as Model
import wandb
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from torchmetrics import MeanMetric


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

    ds = GraspingDataset(Config.Data.dataset_path, 'data/MCD/build_datasets/train_test_dataset.json',
                         subset='holdout_models_holdout_views')
    dl = DataLoader(
        ds,
        shuffle=False,
        batch_size=Config.Eval.mb_size,
        drop_last=False,
        num_workers=Config.General.num_workers,
        pin_memory=True)

    jaccard = MeanMetric().cuda()
    t = trange(len(dl))
    for i, data in zip(t, dl):
        partial, ground_truth = data
        partial, ground_truth = partial.cuda(), ground_truth.cuda()

        reconstruction, probailities = model(partial, num_points=400_000)

        # aux, _ = model(partial, num_points=200_000)
        # point_idx = farthest_point_sampler(aux, 8192 * 2)
        # reconstruction = aux[torch.arange(point_idx.shape[0]).unsqueeze(-1), point_idx]

        grid1 = voxelize_pc(reconstruction, 0.025)
        grid2 = voxelize_pc(ground_truth, 0.025)

        jaccard(torch.sum(grid1 * grid2, dim=[1, 2, 3]) / torch.sum((grid1 + grid2) != 0, dim=[1, 2, 3]))
        t.set_postfix(jaccard=jaccard.compute())
        # o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    main()
