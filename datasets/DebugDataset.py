from pathlib import Path

import open3d
import open3d.cpu.pybind.io
import torch.utils.data as data
import numpy as np
import os, sys

from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
import random
import json


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py


CATEGORY_FILE_PATH = 'data/PCN/PCN.json'
N_POINTS = 16384
N_RENDERINGS = 8
PARTIAL_POINTS_PATH = 'data/PCN/%s/partial/%s/%s/%02d.pcd'
COMPLETE_POINTS_PATH = 'data/PCN/%s/complete/%s/%s.pcd'


class DebugDataset(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, subset):
        with Path(f'data/PCN/{subset}').open('r') as f:
            self.lines = [l.strip() for l in f.readlines()]

    def __getitem__(self, idx):
        ex = self.lines[idx]
        label_id, class_id = ex.split('/')

        complete = str(Path(f'data/PCN/val/complete/{ex}.pcd'))
        partial = str(Path(f'data/PCN/val/partial/{ex}/00.pcd'))

        partial = np.array(open3d.cpu.pybind.io.read_point_cloud(partial).points, dtype=np.float32)
        complete = np.array(open3d.cpu.pybind.io.read_point_cloud(complete).points, dtype=np.float32)

        partial = np.concatenate([partial, np.zeros([2048 - partial.shape[0], 3], dtype=np.float32)])

        return label_id, class_id, (partial, complete)

    def __len__(self):
        return len(self.lines)