import open3d as o3d
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
from pathlib import Path
from torch import nn
from torch.optim import SGD
import numpy as np

# Load a mesh
example = Path('..') / '..' / 'data/ShapeNetCore.v2/02691156/1a9b552befd6306cc8f2d5fe7449af61'
mesh = o3d.io.read_triangle_mesh(str(example / 'models/model_normalized.obj'), False)


# Convert it to point cloud
# complete_pcd = mesh.sample_points_uniformly(22290)
# draw_geometries([complete_pcd])

# Define the implicit function
class ImplicitFunction(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)

        layers = []

        # Input Layer
        layers.append([
            torch.zeros((3, 32), device=device, requires_grad=True),
            torch.zeros((1, 32), device=device, requires_grad=True),
            torch.zeros((1, 32), device=device, requires_grad=True)
        ])

        # Hidden Layers
        for _ in range(2):
            layers.append([
                torch.zeros((32, 32), device=device, requires_grad=True),
                torch.zeros((1, 32), device=device, requires_grad=True),
                torch.zeros((1, 32), device=device, requires_grad=True)
            ])

        layers.append([
            torch.zeros((32, 1), device=device, requires_grad=True),
            torch.zeros((1, 1), device=device, requires_grad=True),
            torch.zeros((1, 1), device=device, requires_grad=True)
        ])

        self.layers = layers
        self._initialize()

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = torch.mm(x, l[0]) * l[1] + l[2]
            x = self.dropout(x)
            x = self.relu(x)

        l = self.layers[-1]
        x = torch.mm(x, l[0]) * l[1] + l[2]

        return x

    def backward(self, loss):
        for l in self.layers:
            l[0].grad = torch.autograd.grad(loss, l[0])
            l[1].grad = torch.autograd.grad(loss, l[1])
            l[2].grad = torch.autograd.grad(loss, l[2])

    def params(self):
        return [param for l in self.layers for param in l]

    def _initialize(self):
        for l in self.layers:
            nn.init.xavier_uniform_(l[0])
            nn.init.xavier_uniform_(l[1])


f = ImplicitFunction('cuda')
optim = SGD(f.params(), lr=0.05)
criterion = torch.nn.BCEWithLogitsLoss()

# Mesh Sampling

def uniform_signed_sampling(mesh, n_points=2048):
    n_uniform = int(n_points * 0.1)
    n_noise = int(n_points * 0.4)
    n_mesh = int(n_points * 0.5)

    points_uniform = np.random.rand(n_uniform, 3) - 0.5

    points_surface = np.array(mesh.sample_points_uniformly(n_mesh).points)
    points_surface = points_surface + (0.1 * np.random.randn(len(points_surface), 3))

    points = np.concatenate([points_uniform, points_surface], axis=0)

    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh)
    query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)

    signed_distance = scene.compute_distance(query_points)
    labels = signed_distance <= 0.001

    return points, labels

# Encode
for e in range(50):
    x, y = uniform_signed_sampling(mesh, n_points=2048*2**7)

    colors = []
    points = []
    for point, label in zip(x, y):
        if label == 1:
            colors.append(np.array([0, 1, 0]))
            points.append(point)
        # else:
        #     colors.append(np.array([1, 0, 0]))
    points, colors = np.stack(points), np.stack(colors)
    points, colors = Vector3dVector(points), Vector3dVector(colors)

    pc = PointCloud()
    pc.points = points
    print(len(points))
    # pc.colors = colors

    draw_geometries([pc])

    x, y, = torch.tensor(x, device='cuda', dtype=torch.float32), torch.tensor(y, device='cuda', dtype=torch.float32)

    out = f(x)
    loss = criterion(out, y)

    optim.zero_grad()
    f.backward(loss)
    optim.step()
