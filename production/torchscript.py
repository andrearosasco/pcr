import numpy as np
import torch

import torch
from polygraphy.common import TensorMetadata
from polygraphy.comparator import DataLoader
import tqdm

from configs import server_config
from model import PCRNetwork


# # Load model
model = PCRNetwork.load_from_checkpoint('./checkpoint/final', config=server_config.ModelConfig)
model.cuda()
model.backbone.cuda()
model.eval()
model.backbone.eval()

x = torch.ones((1, 2024, 3)).cuda()
backbone = torch.jit.trace(model.backbone, x)
backbone.save('pcr.pt')

it = 100
data_loader = DataLoader(iterations=it,
                         val_range=(-0.5, 0.5),
                         input_metadata=TensorMetadata.from_feed_dict(
                             {'input': np.zeros([1, 2024, 3], dtype=np.float32)}))


for x in tqdm.tqdm(data_loader):
    backbone(torch.tensor(x['input']).cuda())
