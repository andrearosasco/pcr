import numpy as np

# generate some random test points
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
from torch import nn


class Ransac(nn.Module):

    def __init__(self, n_points, eps, iterations):
        super().__init__()
        self.n_points = n_points
        self.eps = eps
        self.iterations = iterations

    def forward(self, x):
        # Warning: this way I'm not taking the same point twice. It could be an issue for small pointclouds
        # idxs = torch.randint(low=0, high=x.shape[0], size=self.n_points * self.iterations)
        # subsets = x[idxs]
        # subsets.reshape(self.n_points, self.iterations, 3)


        return self.fit_plane(x)

    def fit_plane(self, points):
        # now find the best-fitting plane for the test points

        # subtract out the centroid and take the SVD
        svd = torch.svd(points - torch.mean(points, dim=1, keepdim=True))

        # Extract the left singular vectors
        left = svd[0]
        left = left[:, -1]

        v = left / torch.norm(left)
        p = - (left.T @ torch.mean(points, dim=1, keepdim=True))

        distance = (points[..., 3] @ v) - p


        return torch.concat([v, p])

def test():
    ransac = Ransac()
    while True:
        m = 20 # number of points
        delta = 0 # size of random displacement
        origin = torch.rand(3, 1) # random origin for the plane
        basis = torch.rand(3, 2) # random basis vectors for the plane
        coefficients = torch.rand(2, m) # random coefficients for points on the plane

        # generate random points on the plane and add random displacement
        points = basis @ coefficients \
                 + torch.tile(origin, (1, m)) \
                 + delta * torch.rand(3, m)

        plane_model = ransac(points)

        aux = PointCloud()
        aux.points = Vector3dVector(points.numpy().T)
        draw_geometries([aux, plot_plane(*plane_model)])

def build():
    ransac = Ransac(1000, 0.01*1.5, 1000)

    m = 20  # number of points
    delta = 0  # size of random displacement
    origin = torch.rand(3, 1)  # random origin for the plane
    basis = torch.rand(3, 2)  # random basis vectors for the plane
    coefficients = torch.rand(2, m)  # random coefficients for points on the plane

    # generate random points on the plane and add random displacement
    points = basis @ coefficients \
             + torch.tile(origin, (1, m)) \
             + delta * torch.rand(3, m)

    torch.onnx.export(ransac, points, './delete.onnx', input_names=['input'],
                      output_names=[f'output'], opset_version=11)

def plot_plane(a, b, c, d):
    xy = np.random.rand(1000, 2)
    z = - ((a * xy[..., 0] + b * xy[..., 1] + d) / c)

    plane = np.concatenate([xy, z[..., None]], axis=1)

    aux = PointCloud()
    aux.points = Vector3dVector(plane)
    return aux


if __name__ == '__main__':
    build()