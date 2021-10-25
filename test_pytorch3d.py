from numpy import cos, sin
from pytorch3d.renderer.mesh import TexturedSoftPhongShader
from torch import nn
from pytorch3d.renderer.mesh.shading import flat_shading
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer import softmax_rgb_blend, HardPhongShader
import torch
from pytorch3d.io import IO
from pytorch3d.io import load_objs_as_meshes, load_obj
import matplotlib.pyplot as plt
import numpy as np


# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import random


def main():
    padding_length = 0

    # Extract point cloud from mesh
    import os
    complete_path = os.path.normpath("C:/Users/sberti/PycharmProjects/pcr/data/ShapeNetCore.v2/03624134/3dcaa6f00e9d91257d28657a5a16b62d/models/model_normalized.obj")

    device = torch.device("cuda:0")
    mesh = load_objs_as_meshes([complete_path], load_textures=True, create_texture_atlas=True, device=device)

    # Normalize mesh
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    # Init rasterizer settings
    dist = random.uniform(1, 2)  # TODO CHECK
    elev = random.uniform(-90, 90)  # TODO CHECK
    azim = random.uniform(0, 360)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=True)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=512, blur_radius=0.0, faces_per_pixel=1
    )

    # Init shader settings
    materials = Materials(device=device)

    # Light
    sph_radius = random.uniform(10, 12)
    y_light = random.uniform(-sph_radius, sph_radius)
    theta = random.uniform(0, 2 * np.pi)
    x_light = np.sqrt(sph_radius ** 2 - y_light ** 2) * cos(theta)
    z_light = np.sqrt(sph_radius ** 2 - y_light ** 2) * sin(theta)
    lights = PointLights(device=device, location=[[x_light, y_light, z_light]])

    # Place light behind the cow in world space. The front of
    # the cow is facing the -z direction.
    lights.location = torch.tensor([0.0, 0.0, 2.0], device=device)[None]

    blend_params = BlendParams(
        sigma=1e-1,
        gamma=1e-4,
        background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
    )
    # Init renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            lights=lights,
            cameras=cameras,
            materials=materials,
            blend_params=blend_params,
        ),
    )
    images = renderer(mesh, cull_backface=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.show()
    return


if __name__ == "__main__":
    while True:
        main()
