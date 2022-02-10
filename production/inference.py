import time

import numpy as np
# import torch
# from configs import ModelConfig
# from model import PCRNetwork
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
from polygraphy.common import TensorMetadata
from polygraphy.comparator import DataLoader

data_loader = DataLoader(iterations=1,
                         val_range=(-0.5, 0.5),
                         input_metadata=TensorMetadata.from_feed_dict(
                             {'input': np.zeros([1, 2024, 3], dtype=np.float32)}))

with open('assets/production/pcr.engine', 'rb') as f:
    serialized_engine = f.read()
    engine = EngineFromBytes(serialized_engine)

tot = 0
for i, x in enumerate(data_loader):
    with TrtRunner(engine) as runner:
        start = time.time()
        outputs = runner.infer(feed_dict=x)

        tot += time.time() - start

print(tot / 100)


#
# model = PCRNetwork.load_from_checkpoint('checkpoint/absolute_best', config=ModelConfig)
# model = model.to('cuda')
# model.eval()
#
# tot = 0
# for i, x in enumerate(data_loader):
#     start = time.time()
#     with torch.no_grad():
#         out = model.backbone(torch.tensor(x['input']).cuda())
#     tot += time.time() - start
#
# print(tot / 100)
