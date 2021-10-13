import open3d as o3d
import trimesh
import tqdm
# try:
#     trimesh.load_mesh('C:/Users/sberti/PycharmProjects/pcr/data/ShapeNetCore.v2/04401088/a4910da0271b6f213a7e932df8806f9e/models/model_normalized.obj')
# except Exception as e:
#     print(str(e))
from torch.utils.data import DataLoader

from datasets.ShapeNetPOV import ShapeNet
import os
import torch
import numpy as np
import random
from configs.cfg1 import DataConfig, TrainConfig

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
seed = TrainConfig.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
g = torch.Generator()
g.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


train_loader = DataLoader(ShapeNet(DataConfig, mode="train"),
                          batch_size=64,
                          shuffle=True,
                          drop_last=True,
                          num_workers=0,
                          pin_memory=True,
                          generator=g)

for elem in tqdm.tqdm(train_loader):
    pass

tm = o3d.io.read_triangle_mesh('C:/Users/sberti/PycharmProjects/pcr/data/ShapeNetCore.v2/04401088/a4910da0271b6f213a7e932df8806f9e/models/model_normalized.obj')
# ..\..\data\ShapeNetCore.v2\03691459\1f2a8562a2de13a2c29fde65e51f52cb
o3d.visualization.draw_geometries([tm])
