from utils.logger import Logger

import os
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from torch import nn
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from datasets.ShapeNetPOV import ShapeNet
from models.HyperNetwork import HyperNetwork
import torch
from configs.local_config import DataConfig, ModelConfig, TrainConfig
from tqdm import tqdm
import copy
from utils.misc import create_3d_grid, check_mesh_contains
import open3d as o3d


def main(test=False):
    logger = Logger(active=True)
    open("bad_files.txt", "w").close()  # Erase previous bad files
    print("Batch size: ", TrainConfig.mb_size)
    print("BackBone input dimension: ", DataConfig.partial_points)
    if TrainConfig.overfit_mode:
        print("ATTENTION: OVERFIT MODE IS ACTIVE!!!!!")
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
    # if "cuda" in TrainConfig.device:
    #     model = nn.DataParallel(model)

    for parameter in model.parameters():
        if len(parameter.size()) > 2:
            torch.nn.init.uniform_(parameter)

    model.to(TrainConfig.device)
    model.train()

    # Loss
    loss_function = TrainConfig.loss(reduction=TrainConfig.loss_reduction)

    # Optimizer
    optimizer = TrainConfig.optimizer(model.parameters(), lr=TrainConfig.lr)

    # WANDB
    logger.log_model(model)
    logger.log_config()

    # Dataset
    train_loader = DataLoader(ShapeNet(DataConfig, mode="train", overfit_mode=TrainConfig.overfit_mode),
                              batch_size=TrainConfig.mb_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=TrainConfig.num_workers,
                              pin_memory=True,
                              generator=g)

    print("Loaded ", len(train_loader), " train instances")

    valid_loader = DataLoader(ShapeNet(DataConfig, mode="valid", overfit_mode=TrainConfig.overfit_mode),
                              batch_size=TrainConfig.mb_size,
                              drop_last=True,
                              num_workers=TrainConfig.num_workers,
                              pin_memory=True)
    print("Loaded ", len(valid_loader), " validation instances")

    losses = []
    accuracies = []
    object_id = None
    padding_lengths = []
    best_val_acc = 0

    for e in range(TrainConfig().n_epoch):
        #########
        # TRAIN #
        #########
        # TODO just created new branch, create one hot vector and add it to global feature
        model.train()
        for idx, (label, partial, data, imp_x, imp_y, padding_length) in enumerate(
                tqdm(train_loader, position=0, leave=True, desc="Epoch " + str(e))):

            padding_lengths.append(padding_length.float().mean().item())
            complete = data.to(TrainConfig().device)
            partial = partial.to(TrainConfig().device)
            x, y = imp_x.to(ModelConfig.device), imp_y.to(ModelConfig.device)

            if ModelConfig.use_object_id:
                object_id = torch.zeros((x.shape[0], DataConfig.n_classes), dtype=torch.float).to(x.device)
                object_id[torch.arange(0, x.shape[0]), label] = 1.

            out = model(partial, x, object_id)
            out = out.squeeze()

            loss_value = loss_function(out, y).sum(dim=-1).mean()  # sum and mean with one value doesn't change anything

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
                                    "train/step": idx + e * len(train_loader),
                                    "train/padding_length": sum(padding_lengths) / len(padding_lengths)})
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
                implicit_function_output_just_true = implicit_function_input[implicit_function_input[:, 3] == 1.]

                logger.log_point_clouds({"complete": complete,
                                         "partial": partial,
                                         "implicit_function_input": implicit_function_input,
                                         "implicit_function_output": implicit_function_output,
                                         "implicit_function_output_just_true": implicit_function_output_just_true})
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
            for idx, (label, partial, mesh) in enumerate(
                    tqdm(valid_loader, position=0, leave=True, desc="Validation " + str(e))):
                partial = partial.to(TrainConfig.device)

                # TODO we are passing 100k points (too much?)
                if ModelConfig.use_object_id:
                    object_id = torch.zeros((x.shape[0], DataConfig.n_classes), dtype=torch.float).to(x.device)
                    object_id[torch.arange(0, x.shape[0]), label] = 1.

                out = model(partial, x, object_id)

                y = check_mesh_contains(mesh, x)  # TODO PARALLELIZE IT
                y = torch.FloatTensor(y).to(out.device)
                loss = loss_function(out, y).sum(dim=-1).mean()
                val_losses.append(loss.item())

                pred = copy.deepcopy(out) > 0.5
                accuracy = (pred == y).sum().item() / torch.numel(pred)  # TODO numpy
                val_accuracies.append(accuracy)
                if test:
                    break

        val_acc = sum(val_accuracies) / len(val_accuracies)
        logger.log_metrics({"validation/loss": sum(val_losses) / len(val_losses),
                            "validation/accuracy": val_acc, "validation/step": e})

        # Save best model
        Path("checkpoint").mkdir(exist_ok=True)
        if TrainConfig.save_ckpt is not None:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), Path("checkpoint") / TrainConfig.save_ckpt)
                print("New best checkpoint saved :D ", val_acc)

        reconstruction = torch.cat((x[0], pred[0]), dim=-1).detach().cpu().numpy()
        original = torch.cat((x[0], y[0]), dim=-1).detach().cpu().numpy()

        reconstruction_kept = reconstruction[reconstruction[..., -1] == 1.]
        original_kept = original[original[..., -1] == 1.]

        total = o3d.io.read_triangle_mesh(mesh[0], False)
        total = total.sample_points_uniformly(10000)
        total = np.array(total.points)

        logger.log_point_clouds({"reconstruction": reconstruction,
                                 "original": original,
                                 "reconstruction_kept": reconstruction_kept,
                                 "original_kept": original_kept,
                                 "total": total})

        if test:
            break


if __name__ == "__main__":
    main()
