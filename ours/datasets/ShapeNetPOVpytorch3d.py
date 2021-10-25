from pathlib import Path
import numpy as np
import pytorch3d.transforms
import torch.utils.data as data
from pytorch3d.renderer import softmax_rgb_blend, SoftPhongShader, SfMPerspectiveCameras, PerspectiveCameras
from pytorch3d.renderer.mesh.shading import flat_shading
from torch import nn
from configs.local_config import TrainConfig
import open3d as o3d
from utils.misc import sample_point_cloud_pytorch3d, fast_from_depth_to_pointcloud
from numpy import cos, sin
from pytorch3d.renderer.blending import BlendParams
import torch
from pytorch3d.io import load_objs_as_meshes
import cv2
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRasterizer,
)
import random
from utils.misc import from_depth_to_pointcloud
from pytorch3d.ops import sample_points_from_meshes


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs):
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


class SoftFlatShader(nn.Module):
    def __init__(
            self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardFlatShader"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = flat_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images


class ShapeNet(data.Dataset):
    def __init__(self, config, mode="train", diff="easy", overfit_mode=False):
        self.mode = mode
        self.overfit_mode = overfit_mode
        #  Backbone Input
        self.data_root = Path(config.dataset_path)
        self.partial_points = config.partial_points
        self.multiplier_complete_sampling = config.multiplier_complete_sampling

        # Implicit function input
        self.noise_rate = config.noise_rate
        self.percentage_sampled = config.percentage_sampled
        self.tollerance = config.tollerance
        self.implicit_input_dimension = config.implicit_input_dimension

        with (self.data_root / diff / f'{self.mode}.txt').open('r') as file:
            lines = file.readlines()

        self.samples = lines
        self.n_samples = len(lines)

        with (self.data_root / 'classes.txt').open('r') as file:
            self.labels_map = {l.split()[1]: l.split()[0] for l in file.readlines()}

    @staticmethod
    def pc_norm(pcs):
        """ pc: NxC, return NxC """
        centroid = np.mean(pcs, axis=0)
        pcs = pcs - centroid
        m = np.max(np.sqrt(np.sum(pcs ** 2, axis=1)))
        pcs = pcs / m
        return pcs

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y
        padding_length = 0

        # Load mesh
        dir_path = self.data_root / self.samples[idx].strip()
        label = int(self.labels_map[dir_path.parent.name])
        complete_path = str(dir_path / 'models/model_normalized.obj')

        if self.overfit_mode:
            complete_path = TrainConfig.overfit_sample

        device = torch.device("cuda:0")
        mesh = load_objs_as_meshes([complete_path], load_textures=True, create_texture_atlas=True, device=device)

        # Normalize mesh to be at the center of a sphere with radius 1
        verts = mesh.verts_packed()
        center_x = (max(verts[:, 0]) + min(verts[:, 0])) / 2
        center_y = (max(verts[:, 1]) + min(verts[:, 1])) / 2
        center_z = (max(verts[:, 2]) + min(verts[:, 2])) / 2
        center = torch.FloatTensor([center_x, center_y, center_z]).cuda()
        mesh.offset_verts_(-center)  # center

        verts = mesh.verts_packed()
        dist = max(torch.sqrt(torch.square(verts[:, 0]) + torch.square(verts[:, 1]) + torch.square(verts[:, 2])))
        mesh.scale_verts_((1.0 / float(dist)))  # scale

        # TODO REMOVE DEBUG
        from pytorch3d.structures import Meshes
        rotation = pytorch3d.transforms.random_rotation().cuda()

        a = (mesh.verts_packed() @ rotation).cpu()
        b = mesh.faces_packed().cpu()

        mesh = mesh.cpu()

        c = mesh.textures

        mesh = Meshes([a], [b], c)

        points = sample_points_from_meshes(mesh)[0]

        # points = points @ rotation

        for i in range(1000):
            sph_radius = 1
            y_light = random.uniform(-sph_radius, sph_radius)
            theta = random.uniform(0, 2 * np.pi)
            x_light = np.sqrt(sph_radius ** 2 - y_light ** 2) * cos(theta)
            z_light = np.sqrt(sph_radius ** 2 - y_light ** 2) * sin(theta)
            points = torch.cat((points, torch.FloatTensor([x_light, y_light, z_light]).unsqueeze(0).to(points.device)))

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        pc = PointCloud()
        pc.points = Vector3dVector(points.cpu())
        o3d.visualization.draw_geometries([pc, coord])
        # TODO END REMOVE DEBUG

        mesh = mesh.cuda()

        # Init rasterizer settings
        dist = 5  # random.uniform(3, 4)
        elev = random.uniform(-90, 90)
        azim = random.uniform(0, 360)
        # R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=True)
        R, T = look_at_view_transform(dist=1.)

        cameras = PerspectiveCameras(
            R=R,
            T=T,
            device=device,
        )

        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=5
        )

        # Init shader settings
        materials = Materials(device=device)

        # Light
        sph_radius = random.uniform(3, 4)
        y_light = random.uniform(-sph_radius, sph_radius)
        theta = random.uniform(0, 2 * np.pi)
        x_light = np.sqrt(sph_radius ** 2 - y_light ** 2) * cos(theta)
        z_light = np.sqrt(sph_radius ** 2 - y_light ** 2) * sin(theta)
        lights = PointLights(device=device, location=[[x_light, y_light, z_light]])
        blend_params = BlendParams(
            sigma=1e-1,
            gamma=1e-4,
            background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
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
        cv2.imshow("RGB", images[0, ..., :3].cpu().numpy())
        cv2.imshow("DEPTH", depth[0, ..., 0].cpu().numpy())

        # Create complete
        complete = sample_points_from_meshes(mesh, self.partial_points)

        # Create partial
        depth = depth[0, ..., 0]
        # depth = (depth - depth.min()) / (depth.max() - depth.min())

        partial = fast_from_depth_to_pointcloud(depth, cameras, R, T)

        # TODO DEBUG START HERE
        pc = PointCloud()
        pc.points = Vector3dVector(partial.cpu())
        o3d.visualization.draw_geometries([pc, coord])
        # TODO END DEBUG

        # Set partial_pcd such that it has the same size of the others
        if partial.shape[0] < self.partial_points:
            diff = self.partial_points - partial.shape[0]
            partial = torch.cat((partial, torch.zeros(diff, 3)))
            padding_length = diff
        else:
            perm = torch.randperm(partial.shape[0])
            ids = perm[:self.partial_points]
            partial = partial[ids]

        # Create implicit function input and output
        imp_x, imp_y = sample_point_cloud_pytorch3d(mesh,
                                                    noise_rate=self.noise_rate,
                                                    percentage_sampled=self.percentage_sampled,
                                                    total=self.implicit_input_dimension,
                                                    tollerance=self.tollerance)

        return label, images, partial, complete, imp_x, imp_y, padding_length

    def __len__(self):
        return int(self.n_samples / (100 if self.overfit_mode else 1))


if __name__ == "__main__":
    from ours.configs.local_config import DataConfig
    from tqdm import tqdm
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector

    a = DataConfig()
    a.dataset_path = Path("..", "..", "data", "ShapeNetCore.v2")
    iterator = ShapeNet(a)
    for elem in tqdm(iterator):
        continue
        lab, image, part, comp, x, y, pad = elem

        # Convert for visualization
        image = image[0, ..., :3].cpu().numpy()
        part = part.cpu().numpy()
        comp = comp.squeeze(0).cpu().numpy()
        x = x.squeeze(0).cpu().numpy()
        y = y.cpu().numpy()

        # Label
        print(lab)

        # Image
        cv2.imshow("Image", image)
        cv2.waitKey()

        # Partial
        pc = PointCloud()
        pc.points = Vector3dVector(comp)
        o3d.visualization.draw_geometries([pc])

        # Complete
        pc = PointCloud()
        pc.points = Vector3dVector(part)
        o3d.visualization.draw_geometries([pc])

        # X & Y
        colors = []
        for e in y:
            if e == 1.:
                colors.append(np.array([0, 1, 0]))
            if e == 0.:
                colors.append(np.array([1, 0, 0]))
        colors = np.stack(colors)
        pc = PointCloud()
        pc.points = Vector3dVector(x)
        pc.colors = Vector3dVector(colors)
        o3d.visualization.draw_geometries([pc])

        # pad
        print("Pad: ", pad)
