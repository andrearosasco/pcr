import os
import time

import numpy as np
import torch
import torch.utils.data as data

from configs.cfg1 import DataConfig
from utils import misc
from utils.misc import sample_point_cloud


class ShapeNet(data.Dataset):
    def __init__(self, config):
        #  Backbone Input
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        # Implicit function input
        self.voxel_size = config.voxel_size
        self.noise_rate = config.noise_rate
        self.percentage_sampled = config.percentage_sampled

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = np.load(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        data = self.pc_norm(data)
        full = torch.from_numpy(data).float()
        # TODO should we have to norm them?
        # They are already between -1/1 but that's true for shapenet points and they are normed
        imp_x, imp_y = sample_point_cloud(full, self.voxel_size,
                                          self.noise_rate,
                                          self.percentage_sampled)
        imp_x, imp_y = torch.tensor(imp_x).float(), torch.tensor(imp_y).float()

        partial, _ = misc.seprate_point_cloud(full.unsqueeze(0), DataConfig.N_POINTS,
                                             [int(DataConfig.N_POINTS * 1 / 4), int(DataConfig.N_POINTS * 3 / 4)],
                                              fixed_points=None)

        return sample['taxonomy_id'], sample['model_id'], partial.squeeze(0), full, imp_x, imp_y

    def __len__(self):
        return len(self.file_list)