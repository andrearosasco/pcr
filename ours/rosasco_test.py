# Read the mesh
from pathlib import Path
import random

import cv2
import numpy as np
import open3d.cpu.pybind.geometry
import pytorch3d
import torch
from numpy import cos, sin
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
import open3d as o3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform, RasterizationSettings, PointLights, \
    BlendParams, Materials, MeshRasterizer, SoftPhongShader
from datasets.ShapeNetPOVpytorch3d import MeshRendererWithDepth
from pytorch3d.structures import Meshes

from configs.local_config import DataConfig, TrainConfig


def draw(pc):
    points = []
    for i in range(1000):
        sph_radius = 1
        y_light = random.uniform(-sph_radius, sph_radius)
        theta = random.uniform(0, 2 * np.pi)
        x_light = np.sqrt(sph_radius * 2 - y_light * 2) * cos(theta)
        z_light = np.sqrt(sph_radius * 2 - y_light * 2) * sin(theta)
        points.append([x_light, y_light, z_light])


    sphere = PointCloud()
    sphere.points = Vector3dVector(points)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])

    # draw_geometries([sphere, coord, pc])



with (Path(DataConfig.dataset_path) / 'easy' / f'train.txt').open('r') as file:
    lines = file.readlines()
samples = lines
n_samples = len(lines)
idx = 1
dir_path = Path(DataConfig.dataset_path) / samples[idx].strip()
complete_path = str(dir_path / 'models/model_normalized.obj')


mesh = load_objs_as_meshes([complete_path], load_textures=True, create_texture_atlas=True, device=TrainConfig.device)

# ======================================================================
# ============================ NORMALIZATION ===========================
# ======================================================================
# Centered
verts = mesh.verts_packed()
# center_x = (max(verts[:, 0]) + min(verts[:, 0])) / 2
# center_y = (max(verts[:, 1]) + min(verts[:, 1])) / 2
# center_z = (max(verts[:, 2]) + min(verts[:, 2])) / 2
# center = torch.FloatTensor([center_x, center_y, center_z]).cuda()
center = torch.tensor(np.mean(verts.cpu().numpy(), axis=0), device=TrainConfig.device)
mesh.offset_verts_(-center)  # center
# Normalized
verts = mesh.verts_packed()
dist = max(torch.sqrt(torch.square(verts[:, 0]) + torch.square(verts[:, 1]) + torch.square(verts[:, 2])))
mesh.scale_verts_((1.0 / float(dist)))  # scale

# TODO print START
complete = sample_points_from_meshes(mesh, 4000)
pc = PointCloud()
pc.points = Vector3dVector(complete.cpu().squeeze().numpy())
draw(pc)
# TODO print END
# ======================================================================
# ============================== ROTATION ==============================
# ======================================================================

rotation = pytorch3d.transforms.random_rotation().cuda()
a = (mesh.verts_packed() @ rotation).cpu()
b = mesh.faces_packed().cpu()
mesh = mesh.cpu()
c = mesh.textures

mesh = Meshes([a], [b], c)

points = sample_points_from_meshes(mesh)[0]

mesh = mesh.cuda()

# TODO print START
complete = sample_points_from_meshes(mesh, 4000)
pc = PointCloud()
pc.points = Vector3dVector(complete.cpu().squeeze().numpy())
draw(pc)
# TODO print END

# ======================================================================
# ============================ VISUALIZATION ===========================
# ======================================================================

R, T = look_at_view_transform(dist=1.5,
                              elev=0.0,
                              azim=0.0)

cameras = PerspectiveCameras(
    # in_ndc=False,
    # image_size=torch.tensor([512, 512]).unsqueeze(0),
    # focal_length=(76.2,),
    # principal_point=((114.8, 31.75),),
    R=R,
    T=T,
    device=TrainConfig.device,
)

raster_settings = RasterizationSettings(
    image_size=512, blur_radius=0.0, faces_per_pixel=5
)

# Init shader settings
materials = Materials(device=TrainConfig.device)

# Light
sph_radius = random.uniform(3, 4)
y_light = random.uniform(-sph_radius, sph_radius)
theta = random.uniform(0, 2 * np.pi)
x_light = np.sqrt(sph_radius * 2 - y_light * 2) * cos(theta)
z_light = np.sqrt(sph_radius * 2 - y_light * 2) * sin(theta)
lights = PointLights(device=TrainConfig.device, location=[[x_light, y_light, z_light]])
blend_params = BlendParams(
    sigma=1e-1,
    gamma=1e-4,
    background_color=torch.tensor([1.0, 1.0, 1.0], device=TrainConfig.device),
)

# Init renderer
renderer = MeshRendererWithDepth(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(
        lights=lights,
        cameras=cameras,
        materials=materials,
        blend_params=blend_params,
    ),
)
images, depth = renderer(mesh, cull_backface=True)

cv2.imshow('depth', depth[0, ..., 0].cpu().numpy())
cv2.imshow('image', images[0, ..., :3].cpu().numpy())
cv2.waitKey(0)

# ======================================================================
# ======================== CREATE POINT CLOUD ==========================
# ======================================================================
# TODO EXPERIMENT
camera = open3d.cpu.pybind.camera.PinholeCameraIntrinsic()
pc = open3d.cpu.pybind.geometry.PointCloud.create_from_depth_image(depth[0, ..., 0].cpu().numpy(), camera)
o3d.visualization.draw_geometries([pc])
# TODO END

rgb_image = images[0, ..., :3].cpu().numpy()
depth_image = depth[0, ..., 0].cpu().numpy()

depth_image = o3d.geometry.Image(depth_image)
rgb_image = o3d.geometry.Image(rgb_image)
rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(rgb_image, depth_image,
                                                            convert_rgb_to_intensity=False,
                                                            depth_scale=1500)

intrinsics = o3d.camera.PinholeCameraIntrinsic()  # 512, 512, 76.2, 76.2, 114.8, 31.75)

pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsics)
pcd.paint_uniform_color([0, 0, 1])
pcd.paint_uniform_color([0, 1, 0])

coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
coord = coord.rotate(rotation.cpu().squeeze().T.numpy(), center=[0, 0, 0])

draw_geometries([pc, pcd, coord])

t1 = cameras.get_projection_transform()
t2 = cameras.get_ndc_camera_transform()
t3 = cameras.get_full_projection_transform()
t4 = cameras.get_world_to_view_transform()

# for t in [t2, t3, t4]:
#     pcd.transform(t.get_matrix().cpu().squeeze().numpy())
#     draw_geometries([pc, pcd, coord])
#     pcd.transform(t.get_matrix().T.cpu().squeeze().numpy())
#     draw_geometries([pc, pcd, coord])
#     pcd.transform(torch.inverse(t.get_matrix()).cpu().squeeze().numpy())
#     draw_geometries([pc, pcd, coord])
#     pcd.transform(torch.inverse(t.get_matrix()).T.cpu().squeeze().numpy())
#     draw_geometries([pc, pcd, coord])