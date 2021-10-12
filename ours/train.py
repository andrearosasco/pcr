import os
import random
import numpy as np
from torch import nn
from torch.nn.utils import clip_grad_value_
import open3d as o3d
from torch.utils.data import DataLoader
from datasets.ShapeNetPOV import ShapeNet
from models.HyperNetwork import HyperNetwork
import torch
from configs.cfg1 import DataConfig, ModelConfig, TrainConfig
from tqdm import tqdm
import copy
from utils.logger import Logger
from utils.misc import create_3d_grid, check_mesh_contains


def main(test=False):
    print("Batch size: ", TrainConfig.mb_size)
    print("BackBone input dimension: ", DataConfig.partial_points)
    os.environ['CUDA_VISIBLE_DEVICES'] = TrainConfig.visible_dev

    # Reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed = TrainConfig.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # Model
    model = HyperNetwork(ModelConfig())
    model = nn.DataParallel(model)

    for parameter in model.parameters():
        if len(parameter.size()) > 2:
            torch.nn.init.uniform_(parameter)

    model.to(TrainConfig.device)
    model.train()

    # Loss
    loss_function = TrainConfig.loss(reduction="none")

    # Optimizer
    optimizer = TrainConfig.optimizer(model.parameters())

    # WANDB
    logger = Logger(model, active=True)

    # Dataset
    train_loader = DataLoader(ShapeNet(DataConfig, mode="train"),
                              batch_size=TrainConfig.mb_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=TrainConfig.num_workers,
                              pin_memory=True,
                              generator=g)
    print("Loaded ", len(train_loader), " train instances")

    valid_loader = DataLoader(ShapeNet(DataConfig, mode="valid"),
                              batch_size=TrainConfig.mb_size,
                              drop_last=True,
                              num_workers=TrainConfig.num_workers,
                              pin_memory=True)
    print("Loaded ", len(train_loader), " validation instances")

    losses = []
    accuracies = []

    for e in range(TrainConfig().n_epoch):
        #########
        # TRAIN #
        #########
        model.train()
        for idx, (label, partial, data, imp_x, imp_y) in enumerate(
                tqdm(train_loader, position=0, leave=True, desc="Epoch " + str(e))):

            complete = data.to(TrainConfig().device)
            partial = partial.to(TrainConfig().device)
            x, y = imp_x.to(ModelConfig.device), imp_y.to(ModelConfig.device)

            out = model(partial, x)
            out = out.squeeze()

            loss_value = loss_function(out, y).sum(dim=1).mean()

            optimizer.zero_grad()
            loss_value.backward()
            if TrainConfig.clip_value is not None:
                clip_grad_value_(model.parameters(), TrainConfig.clip_value)

            optimizer.step()

            # Logs
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()[..., None]
            out = out.detach().cpu().numpy()[..., None]
            loss_value = loss_value.item()

            losses.append(loss_value)
            pred = copy.deepcopy(out) > 0.5
            accuracy = (pred == y).sum() / pred.size
            accuracies.append(accuracy)

            if idx % TrainConfig.log_metrics_every == 0:  # Log numerical stuff
                logger.log_metrics({"train/loss": sum(losses) / len(losses),
                                    "train/accuracy": sum(accuracies) / len(accuracies),
                                    "train/out": out,
                                    "train/step": idx})
                losses = []
                accuracies = []

            if idx % TrainConfig.log_pcs_every == 0:  # Log point clouds

                complete = complete.detach().cpu().numpy()[0]
                partial = partial.detach().cpu().numpy()[0]
                pred = pred[0]
                x = x[0]
                y = y[0]

                implicit_function_input = np.concatenate((x, y), axis=1)
                implicit_function_output = np.concatenate((x, pred), axis=1)

                logger.log_point_clouds({"complete": complete,
                                         "partial": partial,
                                         "implicit_function_input": implicit_function_input,
                                         "implicit_function_output": implicit_function_output})

            if test:
                break
        ########
        # EVAL #
        ########
        x = create_3d_grid(bs=TrainConfig().mb_size).to(TrainConfig().device)
        pred = None
        val_losses = []
        val_accuracies = []
        model.eval()
        with torch.no_grad():
            for idx, (label, mesh, partial) in enumerate(
                    tqdm(valid_loader, position=0, leave=True, desc="Validation " + str(e))):
                partial = partial.to(TrainConfig.device)

                out = model(partial, x)

                y = check_mesh_contains(mesh, x)  # TODO PARALLELIZE IT
                y = torch.FloatTensor(y).to(out.device)
                loss = loss_function(out, y).sum(dim=1).mean()
                val_losses.append(loss.item())

                pred = copy.deepcopy(out) > 0.5
                accuracy = (pred == y).sum().item() / torch.numel(pred)  # TODO numpy
                val_accuracies.append(accuracy)

                if test:
                    break

        logger.log_metrics({"validation/loss": sum(val_losses) / len(val_losses),
                            "validation/accuracy": sum(val_accuracies) / len(val_accuracies)})
        reconstruction = torch.cat((x[0], pred[0]), dim=-1).detach().cpu().numpy()
        original = torch.cat((x[0], y[0]), dim=-1).detach().cpu().numpy()
        logger.log_point_clouds({"reconstruction": reconstruction,
                                 "original": original})

        if test:
            break


if __name__ == "__main__":
    main()
