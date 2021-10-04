from torch.nn import BCELoss, Sigmoid
from torch.utils.data import DataLoader
from torchviz import make_dot

from datasets.ShapeNet55Dataset import ShapeNet
from models.HyperNetwork import BackBone, ImplicitFunction, HyperNetwork
import torch
from utils import misc
import time
from configs.cfg1 import DataConfig, ModelConfig, TrainConfig
from utils.misc import sample_point_cloud
from tqdm import tqdm
import copy
from utils.logger import Logger

if __name__ == '__main__':
    # Load Dataset
    dataset = ShapeNet(DataConfig())

    # Model
    model = HyperNetwork(ModelConfig())

    for parameter in model.parameters():
        if len(parameter.size()) > 2:
            torch.nn.init.xavier_uniform_(parameter)
    model.to(TrainConfig().device)
    model.train()

    # Loss
    loss = BCELoss(reduction='none')
    activation = Sigmoid()

    # Optimizer
    # TODO add class to config (e.g. TrainConfig.oprimizer(params=model.parameters()))
    optimizer = torch.optim.Adam(params=model.parameters())

    # WANDB
    logger = Logger(model, active=False)

    # Dataset
    config = TrainConfig
    # TODO: come aggiunge point cloud di dimensioni diverse nella stessa batch?
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True,
                            drop_last=True,
                            num_workers=10, pin_memory=True)

    losses = []
    accuracy = []

    for e in range(TrainConfig().n_epoch):
        for idx, (taxonomy_ids, model_ids, data, imp_x, imp_y) in enumerate(
                tqdm(dataloader, position=0, leave=True, desc="Epoch " + str(e))):

            gt = data.to(TrainConfig().device)
            x, y = imp_x.to(ModelConfig.device), imp_y.to(ModelConfig.device)

            partial, _ = misc.seprate_point_cloud(gt, DataConfig().N_POINTS,
                                                  [int(DataConfig().N_POINTS * 1 / 4), int(DataConfig().N_POINTS * 3 / 4)],
                                                  fixed_points=None)
            partial = partial.cuda()
            start = time.time()

            logits = model(partial, x)

            prob = activation(logits).squeeze(-1)
            loss_value = loss(prob, y).sum(dim=1).mean()

            if idx == 1:
                make_dot(loss_value, params=dict(model.named_parameters())).render("second_call", format="png")

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            end = time.time()

            # Logs
            with torch.no_grad():

                x = x.detach()
                y = y.detach()
                prob = prob.detach()

                losses.append(loss_value.item())
                pred = copy.deepcopy(prob) > 0.5
                accuracy.append(torch.sum((pred == y)) / torch.numel(pred))

                if idx % TrainConfig().log_metrics_every == 0:  # Log numerical stuff
                    logger.log_metrics(losses, accuracy, pred, prob, y)
                    losses = []
                    accuracy = []

                if idx % TrainConfig().log_pcs_every == 0:  # Log point clouds
                    true_points = torch.cat((x[0], y[0].unsqueeze(-1)), dim=1)

                    true = []

                    for point, value in zip(x[0], pred[0]):
                        if value.item() == 1.:
                            true.append(point)
                    if len(true) > 0:
                        true = torch.stack(true)
                    else:
                        true = torch.zeros(1, 3)

                    logger.log_pcs(gt[0], partial[0], true_points, true)
