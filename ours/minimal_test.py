from torch.nn import BCELoss, Sigmoid
from datasets.ShapeNet55Dataset import ShapeNet
from models.PoinTr import Hypernetwork, ImplicitFunction
import torch
from utils import misc
import time
from utility import DataConfig, ModelConfig, TrainConfig, sample_point_cloud, crop_ratio
import wandb
from tqdm import tqdm

# Load Dataset
dataset = ShapeNet(DataConfig())

# Load Model
model = Hypernetwork(ModelConfig())
model.to(TrainConfig().device)

# Loss
loss = BCELoss()
m = Sigmoid()

# Optimizer
optimizer = torch.optim.Adam(params=model.parameters())

# WANDB
wandb.login(key="f5f77cf17fad38aaa2db860576eee24bde163b7a")
wandb.init(project='pcr', entity='stefanoberti')
wandb.config = DataConfig()  # TODO find a better way
wandb.watch(model)

for e in range(ModelConfig().n_epoch):
    for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(dataset, position=0, leave=True, desc="Epoch "+str(e))):
        # taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
        # model_id = model_ids[0]

        gt = data.to(TrainConfig().device)
        choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]), torch.Tensor([-1, 1, 1]),
                  torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
                  torch.Tensor([-1, -1, -1])]
        num_crop = int(DataConfig().N_POINTS * crop_ratio[TrainConfig().difficulty])

        x, y = sample_point_cloud(data, TrainConfig().voxel_size,
                                  TrainConfig().noise_rate,
                                  TrainConfig().percentage_sampled)
        x, y = torch.tensor(x).to(TrainConfig().device).float(), torch.tensor(y).to(TrainConfig().device)

        for item in choice:
            partial, _ = misc.seprate_point_cloud(gt.unsqueeze(0), DataConfig().N_POINTS, num_crop, fixed_points=item)
            partial = misc.fps(partial, ModelConfig().PC_SIZE)

            start = time.time()

            optimizer.zero_grad()
            ret = model(partial)
            giulio_l_implicit_function = ImplicitFunction(ret)
            pred = giulio_l_implicit_function(x)
            loss_value = loss(m(pred).squeeze(), y.float())
            loss_value.backward()
            optimizer.step()

            end = time.time()

            wandb.log({"loss": loss_value.item()})
            wandb.log({"hypernetwork_parameters": ret})
            wandb.log({"predicted": m(pred)})
            wandb.log({"true": y.float()})

            coarse_points = ret[0]
            dense_points = ret[1]

            # draw_point_cloud(gt)
            # draw_point_cloud(partial)
            # draw_point_cloud(coarse_points)
            # draw_point_cloud(dense_points)
