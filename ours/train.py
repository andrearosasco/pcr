from torch.nn import BCELoss, Sigmoid
from torch.utils.data import DataLoader
from datasets.ShapeNet55Dataset import ShapeNet
from models.HyperNetwork import BackBone, ImplicitFunction
import torch
from utils import misc
import time
from utility import DataConfig, ModelConfig, TrainConfig, sample_point_cloud, crop_ratio
from tqdm import tqdm
import copy
from logger import Logger

# Load Dataset
dataset = ShapeNet(DataConfig())

# Model
model = BackBone(ModelConfig())
for parameter in model.parameters():
    if len(parameter.size()) > 2:
        torch.nn.init.xavier_uniform_(parameter)
model.to(TrainConfig().device)
model.train()

# Loss
loss = BCELoss()
m = Sigmoid()

# Optimizer
optimizer = torch.optim.Adam(params=model.parameters())

# WANDB
logger = Logger(model)

# Dataset
config = TrainConfig
dataloader = DataLoader(dataset, batch_size=64,
                        shuffle=True,
                        drop_last=True,
                        num_workers=10)

losses = []
accuracies = []
for e in range(TrainConfig().n_epoch):
    for idx, (taxonomy_ids, model_ids, data) in enumerate(
            tqdm(dataset, position=0, leave=True, desc="Epoch " + str(e))):
        # taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
        # model_id = model_ids[0]

        num_crop = int(DataConfig().N_POINTS * crop_ratio[TrainConfig().difficulty])

        x, y = sample_point_cloud(data, TrainConfig().voxel_size,
                                  TrainConfig().noise_rate,
                                  TrainConfig().percentage_sampled)
        x, y = torch.tensor(x).to(TrainConfig().device).float(), torch.tensor(y).to(TrainConfig().device).float()

        gt = data.to(TrainConfig().device)
        partial, _ = misc.seprate_point_cloud(gt.unsqueeze(0), DataConfig().N_POINTS,
                                              [int(DataConfig().N_POINTS * 1 / 4), int(DataConfig().N_POINTS * 3 / 4)],
                                              fixed_points=None)
        partial = partial.cuda()

        start = time.time()

        optimizer.zero_grad()
        ret, z = model(partial)
        giulio_l_implicit_function = ImplicitFunction(ret)
        pred = giulio_l_implicit_function(x)
        y_ = m(pred).squeeze()
        loss_value = loss(y_.unsqueeze(0), y.unsqueeze(0))
        loss_value.backward()
        optimizer.step()

        end = time.time()

        # Logs
        x = x.detach()
        y = y.detach()
        y_ = y_.detach()

        losses.append(loss_value.item())
        pred = copy.deepcopy(y_.cpu()).apply_(lambda v: 1 if v > 0.5 else 0).to(TrainConfig().device)
        accuracy = torch.sum((pred == y)) / torch.numel(pred)
        accuracies.append(accuracy.item())

        if idx % TrainConfig().log_metrics_every == 0:  # Log numerical stuff
            logger.log_metrics(losses, accuracies, pred, y_, y, z, giulio_l_implicit_function.params)
            losses = []
            accuracies = []

        if idx % TrainConfig().log_pcs_every == 0:  # Log point clouds
            true_points = torch.cat((x.detach(), y.unsqueeze(1)), dim=1)

            true = []
            for point, value in zip(x, pred.unsqueeze(1)):
                if value.item() == 1.:
                    true.append(point)
            if len(true) > 0:
                true = torch.stack(true)
            else:
                true = torch.zeros(1, 3)

            logger.log_pcs(gt, partial.squeeze(), true_points, true)
