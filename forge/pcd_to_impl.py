import open3d as o3d
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

import wandb
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np

wandb.login(key="f5f77cf17fad38aaa2db860576eee24bde163b7a")
wandb.init(project='implicit_function', entity='coredump')

# Load a mesh
example = Path('..') / '..' / 'data/ShapeNetCore.v2/02747177/1ce689a5c781af1bcf01bc59d215f0'
mesh = o3d.io.read_triangle_mesh(str(example / 'models/model_normalized.obj'), False)


# Convert it to point cloud
# complete_pcd = mesh.sample_points_uniformly(22290)
# draw_geometries([complete_pcd])

# Define the implicit function
class ImplicitFunction:
    def __init__(self, device):
        super().__init__()
        self.train = True
        self.relu = nn.LeakyReLU(0.2)

        layers = []

        # Using scales the gradient is around 1e-7
        # Without scales the gradient is around 1e-5

        # Input Layer
        layers.append([
            torch.zeros((3, 512), device=device, requires_grad=True),
            # torch.zeros((1, 64), device=device, requires_grad=True),
            torch.zeros((1, 512), device=device, requires_grad=True)
        ])

        # Hidden Layers
        for _ in range(4):
            layers.append([
                torch.zeros((512, 512), device=device, requires_grad=True),
                # torch.zeros((1, 64), device=device, requires_grad=True),
                torch.zeros((1, 512), device=device, requires_grad=True)
            ])

        layers.append([
            torch.zeros((512, 1), device=device, requires_grad=True),
            # torch.zeros((1, 1), device=device, requires_grad=True),
            torch.zeros((1, 1), device=device, requires_grad=True)
        ])

        self.layers = layers
        self._initialize()

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = torch.mm(x, l[0]) + l[1]
            x = F.dropout(x, p=0.5, training=self.train)
            x = self.relu(x)

        l = self.layers[-1]
        x = torch.mm(x, l[0]) + l[1]

        return x

    def eval(self):
        self.train = False

    def train(self, **kwargs):
        self.train = True

    def backward(self, loss):
        # print('==================================== Zero Tensor ==============================================')

        for l in self.layers:
            l[0].grad, = torch.autograd.grad(loss, l[0], only_inputs=True, retain_graph=True)
            # l[1].grad, = torch.autograd.grad(loss, l[1], only_inputs=True, retain_graph=True)
            l[1].grad, = torch.autograd.grad(loss, l[1], only_inputs=True, retain_graph=True)

        # print('==================================== Grad Tensor ==============================================')
        # np.mean([p.grad.cpu().numpy() for p in self.layers[0]])
        # print(np.vstack([p.grad.cpu().numpy() for p in self.layers[0]]).mean())
        # print(np.vstack([p.grad.cpu().numpy() for p in self.layers[1]]).mean())
        # print(np.vstack([p.grad.cpu().numpy() for p in self.layers[2]]).mean())
        # print(np.vstack([p.grad.cpu().numpy() for p in self.layers[3]]).mean())

    def params(self):
        return [param for l in self.layers for param in l]

    def _initialize(self):
        for l in self.layers:
            nn.init.xavier_uniform_(l[0])
            # nn.init.xavier_uniform_(l[1])


f = ImplicitFunction('cuda')
optim = SGD(f.params(), lr=0.1, momentum=0.9)
scheduler = MultiStepLR(optim, milestones=[10000], gamma=0.01)
# scheduler = ReduceLROnPlateau(optim, 'min')
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
activation = nn.Sigmoid()
# Mesh Sampling

def andreas_sampling(mesh, n_points=2048):
    n_uniform = int(n_points * 0.1)
    n_noise = int(n_points * 0.4)
    n_mesh = int(n_points * 0.5)

    points_uniform = np.random.rand(n_uniform, 3) * 2 - 1
    points_noisy = np.array(mesh.sample_points_uniformly(n_noise).points) + (0.1 * np.random.randn(n_noise, 3))
    points_surface = np.array(mesh.sample_points_uniformly(n_mesh).points)

    points = np.concatenate([points_uniform, points_noisy, points_surface], axis=0)

    labels = [False] * (n_uniform + n_noise) + [True] * n_mesh
    return points, labels

# TODO too few positive examples
# def uniform_signed_sampling(mesh, n_points=2048):
#     n_uniform = int(n_points * 0.1)
#     n_mesh = int(n_points * 0.9)
#
#     points_uniform = np.random.rand(n_uniform, 3) * 2 - 1
#     points_surface = np.array(mesh.sample_points_uniformly(n_mesh).points) + (0.1 * np.random.randn(n_mesh, 3))
#
#     points = np.concatenate([points_uniform, points_surface], axis=0)
#
#    scene = o3d.t.geometry.RaycastingScene()
#     mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
#     _ = scene.add_triangles(mesh)
#     query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
#
#     unsigned_distance = scene.compute_distance(query_points)
#     occupancies1 = -0.01 < unsigned_distance
#     occupancies2 = unsigned_distance < 0.01
#     labels = occupancies1 & occupancies2
#
#     return points, labels.numpy()

# Encode
# x, y = uniform_signed_sampling(mesh, n_points=8192)
# x, y, = torch.tensor(x, device='cuda', dtype=torch.float32), torch.tensor(y, device='cuda', dtype=torch.float32)

for e in range(10000):
    x, y = uniform_signed_sampling(mesh, n_points=8192)
    x, y, = torch.tensor(x, device='cuda', dtype=torch.float32), torch.tensor(y, device='cuda', dtype=torch.float32)

    out = f(x)
    loss = criterion(out.squeeze(), y)

    if e in [9998, 9999, 10000, 10001]:
        print(optim.state_dict())

    optim.zero_grad()
    f.backward(loss)
    optim.step()
    scheduler.step()

    wandb.log({
        'train/loss': loss.detach().cpu(),
        'train/accuracy': torch.mean(((activation(out).detach().cpu() > 0.5).squeeze() == y.detach().cpu().bool()).float()),
        'train/step': e
    })


x_l, y_l, o_l = [], [], []
f.eval()
for e in range(10):
    x, y = uniform_signed_sampling(mesh, n_points=2048)
    x, y = torch.tensor(x, device='cuda', dtype=torch.float32), torch.tensor(y, device='cuda', dtype=torch.float32)

    out = f(x)
    loss = criterion(out.squeeze(), y)

    x_l.extend(x.detach().cpu().tolist())
    y_l.extend(y.detach().cpu().tolist())
    o_l.extend(out.detach().cpu().tolist())

threshold = [0.5, 0.8, 0.9, 0.95, 0.99]
pcs = {}
for t in threshold:

    points, classes = [], []
    for point, out, label in zip(x_l, o_l, y_l):
        pred = (activation(out) > t).detach().cpu()

        if pred == 1:
            if label == 1:
                classes.append(1)
            else:
                classes.append(0)
            points.append(point)

    pcs[f'thresh{t}'] = np.concatenate([np.array(points), np.expand_dims(np.array(classes), axis=1)], 1)
        # else:
        #     colors.append(np.array([1, 0, 0]))
        #     classes.append(2)
        #     points.append(point)

    wandb.log({
        'valid/loss': loss.detach().cpu(),
        'valid/accuracy': torch.mean(
            ((activation(out).detach().cpu() > 0.5).squeeze() == y.detach().cpu().bool()).float()),
        'valid/step': e
    })
# scene = o3d.t.geometry.RaycastingScene()
# mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
# _ = scene.add_triangles(mesh)
# query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
# unsigned_distance = scene.compute_distance(query_points)


points, colors = np.stack(points)
points, colors = Vector3dVector(points), Vector3dVector(colors)


pc = PointCloud()
pc.points = points
pc.colors = colors


pcs = wandb.Object3D({"type": "lidar/beta", "points": np.concatenate([np.array(points), np.expand_dims(np.array(classes), axis=1)], 1)})
wandb.log({'valid/pc': pcs})


