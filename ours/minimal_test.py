from torch.nn import BCELoss, Sigmoid
from datasets.ShapeNet55Dataset import ShapeNet
from models.HyperNetwork import BackBone, ImplicitFunction
import torch
from utils import misc
import time
from utility import DataConfig, ModelConfig, TrainConfig, sample_point_cloud, crop_ratio
import wandb
from tqdm import tqdm
import open3d as o3d

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
wandb.login(key="f5f77cf17fad38aaa2db860576eee24bde163b7a")
wandb.init(project='pcr', entity='coredump')
# transform class into dict to log it to wandb
wandb.config["train"] = {k: dict(TrainConfig.__dict__)[k] for k in dict(TrainConfig.__dict__) if not k.startswith("__")}
wandb.config["model"] = {k: dict(ModelConfig.__dict__)[k] for k in dict(ModelConfig.__dict__) if not k.startswith("__")}
wandb.config["data"] = {k: dict(DataConfig.__dict__)[k] for k in dict(DataConfig.__dict__) if not k.startswith("__")}
wandb.watch(model, log="all", log_freq=1, log_graph=True)

for e in range(TrainConfig().n_epoch):
    for idx, (taxonomy_ids, model_ids, data) in enumerate(
            tqdm(dataset, position=0, leave=True, desc="Epoch " + str(e))):
        # taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
        # model_id = model_ids[0]

        gt = data.to(TrainConfig().device)
        # choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]), torch.Tensor([-1, 1, 1]),
        #           torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
        #           torch.Tensor([-1, -1, -1])]  # TODO READD
        choice = [torch.Tensor([1, 1, 1])]  # TODO REMOVE
        num_crop = int(DataConfig().N_POINTS * crop_ratio[TrainConfig().difficulty])

        x, y = sample_point_cloud(data, TrainConfig().voxel_size,
                                  TrainConfig().noise_rate,
                                  TrainConfig().percentage_sampled)
        x, y = torch.tensor(x).to(TrainConfig().device).float(), torch.tensor(y).to(TrainConfig().device).float()

        for item in choice:
            partial, _ = misc.seprate_point_cloud(gt.unsqueeze(0), DataConfig().N_POINTS, num_crop, fixed_points=item)
            partial = misc.fps(partial, ModelConfig().PC_SIZE)

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

            # TODO wandb logs
            if idx % 100 == 0:
                y_ = y_.detach().cpu()
                y = y.detach().cpu()
                y_.apply_(lambda v: 1 if v > 0.5 else 0)
                accuracy = torch.sum(torch.tensor(y_ == y)) / torch.numel(y)
                wandb.log({"accuracy": accuracy.item()})
                wandb.log({"loss": loss_value.item()})
                wandb.log({"hypernetwork_parameters": ret})
                wandb.log({"y_": y_})
                wandb.log({"true": y.float()})
                wandb.log({"predicted": pred})
                wandb.log({"z": z})

                weights = []
                scales = []
                biases = []
                for param in giulio_l_implicit_function.params:
                    weights.append(param[0].view(-1))
                    scales.append(param[1].view(-1))
                    biases.append(param[2].view(-1))
                wandb.log({"w of implicit function": torch.cat(weights)})
                wandb.log({"scales of implicit function": torch.cat(scales)})
                wandb.log({"b of implicit function": torch.cat(biases)})

            if idx % 1000 == 0:
                true_points = torch.cat((x.detach().cpu(), y.unsqueeze(1)), dim=1)

                true = []
                for point, value in zip(x.detach().cpu(), y_.unsqueeze(1)):
                    if value.item() == 1.:
                        true.append(point)
                if len(true) > 0:
                    true = torch.cat(true).reshape(-1, 3)

                wandb.log({"original":
                    wandb.Object3D({
                        "type": "lidar/beta",
                        "points": gt.detach().cpu().numpy()},
                    )}
                )
                wandb.log({"partial":
                    wandb.Object3D({
                        "type": "lidar/beta",
                        "points": partial.squeeze().detach().cpu().numpy()},
                    )}
                )
                wandb.log({"implicit function input":
                    wandb.Object3D({
                        "type": "lidar/beta",
                        "points": true_points.detach().cpu().numpy()},
                    )}
                )
                wandb.log({"predicted point cloud":
                    wandb.Object3D({
                        "type": "lidar/beta",
                        "points": true.detach().cpu().numpy()},
                    )}
                )

            # vis.add_geometry(cube_mesh)
