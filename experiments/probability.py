import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm

import numpy as np
import torch
import wandb
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries

from configs.debug_config import Config
from datasets.GraspingDataset import GraspingDataset
from model.PCRNetwork2 import PCRNetwork as Model
from utils.misc import create_cube


def load_checkpoint(project, id, version):
    ckpt = f'model-{id}:{version}'

    ckpt_path = f'artifacts/{ckpt}/model.ckpt' if os.name != 'nt' else \
        f'artifacts/{ckpt}/model.ckpt'.replace(':', '-')

    if not Path(ckpt_path).exists():
        run = wandb.init(id=id, settings=wandb.Settings(start_method="spawn"))
        run.use_artifact(f'rosasco/{project}/{ckpt}', type='model').download(f'artifacts/{ckpt}/')
        wandb.finish(exit_code=0)

    model = Model.load_from_checkpoint(ckpt_path, config=Config.Model)
    return model


if __name__ == '__main__':
    id = '1g3iwlqy'
    version = 'v22'
    project = 'pcr-grasping'

    model = load_checkpoint(project, id, version)
    model.cuda()
    ds = GraspingDataset(Config.Data.dataset_path, 'data/MCD/build_datasets/train_test_dataset.json',
                         subset='holdout_models_holdout_views')

    # 64059 rugby
    # 55096 torus
    # 123135 ycb bad
    for idx in np.random.permutation(len(ds)):
        print(idx)

        data = ds[123135]

        partial, complete = data
        partial, complete = torch.tensor(partial, device='cuda').unsqueeze(0), torch.tensor(complete, device='cuda').unsqueeze(0)

        pred, prob = model(partial)

        #  Visualization
        norm = mpl.colors.Normalize(vmin=prob.min(), vmax=prob.max())
        m = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        colors = m.to_rgba(prob.cpu().squeeze())[..., :-1]

        o3d_partial = PointCloud(points=Vector3dVector(partial.cpu().squeeze().numpy())) #  + np.array([1, 0, 0])
        o3d_partial.paint_uniform_color([1, 0, 0])
        o3d_complete = PointCloud(points=Vector3dVector(complete.cpu().squeeze().numpy()))
        o3d_complete.paint_uniform_color([1, 1, 0])
        o3d_pred = PointCloud(points=Vector3dVector(pred.cpu().squeeze().numpy()))
        o3d_pred.colors = Vector3dVector(colors)

        # draw_geometries([partial])
        # draw_geometries([pred])
        # draw_geometries([complete])

        # draw_geometries([pred, complete])
        draw_geometries([create_cube(), o3d_complete, o3d_partial])
        # partial = PointCloud(points=Vector3dVector(partial.cpu().squeeze().numpy()))
        # partial.paint_uniform_color([1, 0, 0])
        # complete = PointCloud(points=Vector3dVector(complete.cpu().squeeze().numpy()))
        # complete.paint_uniform_color([1, 1, 0])
        # pred = PointCloud(points=Vector3dVector(pred.cpu().squeeze().numpy()))
        # pred.colors = Vector3dVector(colors)
        # draw_geometries([pred, complete, partial])

