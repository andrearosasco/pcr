from torch.nn import BCELoss, Sigmoid
from datasets.ShapeNet55Dataset import ShapeNet
from models.PoinTr import Hypernetwork, ImplicitFunction
import torch
from utils import misc
import time
from utility import DataConfig, ModelConfig, TrainConfig, sample_point_cloud, crop_ratio

# Load Dataset
dataset = ShapeNet(DataConfig())

# Load Model
model = Hypernetwork(ModelConfig())
model.to(TrainConfig().device)

# Loss
loss = BCELoss()
m = Sigmoid()

with torch.no_grad():
    for idx, (taxonomy_ids, model_ids, data) in enumerate(dataset):
        # taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
        # model_id = model_ids[0]

        gt = data.to(TrainConfig().device)
        choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]), torch.Tensor([-1, 1, 1]),
                  torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
                  torch.Tensor([-1, -1, -1])]
        num_crop = int(DataConfig().N_POINTS * crop_ratio[TrainConfig().difficulty])

        x, y = sample_point_cloud(data)
        x, y = torch.tensor(x).to(TrainConfig().device).float(), torch.tensor(y).to(TrainConfig().device)

        for item in choice:
            partial, _ = misc.seprate_point_cloud(gt.unsqueeze(0), DataConfig().N_POINTS, num_crop, fixed_points=item)
            partial = misc.fps(partial, 2048)

            start = time.time()
            # partial = torch.tensor(partial, requires_grad=True)
            ret = model(partial)
            giulio_l_implicit_function = ImplicitFunction(ret)

            pred = giulio_l_implicit_function(x)
            loss(m(pred).squeeze(), y.float())  # .backward()

            end = time.time()

            print(end-start)

            coarse_points = ret[0]
            dense_points = ret[1]

            # draw_point_cloud(gt)
            # draw_point_cloud(partial)
            # draw_point_cloud(coarse_points)
            # draw_point_cloud(dense_points)
