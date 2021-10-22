from pathlib import Path
import numpy as np
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

        # Extract point cloud from mesh
        dir_path = self.data_root / self.samples[idx].strip()
        label = int(self.labels_map[dir_path.parent.name])
        complete_path = str(dir_path / 'models/model_normalized.obj')

        if self.overfit_mode:
            complete_path = TrainConfig.overfit_sample

        # Load mesh
        device = torch.device("cuda:0")
        mesh = load_objs_as_meshes([complete_path], load_textures=True, create_texture_atlas=True, device=device)

        # Normalize mesh
        verts = mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))

        # Init rasterizer settings
        dist = random.uniform(3, 4)  # TODO CHECK
        elev = random.uniform(-90, 90)  # TODO CHECK
        azim = random.uniform(0, 360)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=True)

        # K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])

        cameras = FoVPerspectiveCameras(
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
        lights.location = torch.tensor([0.0, 0.0, 2.0], device=device)[None]
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

        # TODO REMOVE DEBUG
        # cv2.imshow("Image", images[0, :, :, :3].cpu().numpy())
        # cv2.waitKey()
        # TODO REMOVE DEBUG

        # Create complete
        complete = sample_points_from_meshes(mesh, self.partial_points)

        # Create partial
        depth = depth[0, ..., 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = depth

        # TODO REMOVE
        # xyz = complete
        # # transform xyz to the camera view coordinates
        # xyz_cam = cameras.get_world_to_view_transform().transform_points(xyz)
        # # extract the depth of each point as the 3rd coord of xyz_cam
        # depth = xyz_cam[:, :, 2:]
        # # project the points xyz to the camera
        # xy = cameras.transform_points(xyz)[:, :, :2]
        # # append depth to xy
        # xy_depth = torch.cat((xy, depth), dim=2)
        # # unproject to the world coordinates
        # xyz_unproj_world = cameras.unproject_points(xy_depth, world_coordinates=True)
        # print(torch.allclose(xyz, xyz_unproj_world))  # True
        # # unproject to the camera coordinates
        # xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)
        # print(torch.allclose(xyz_cam, xyz_unproj))  # True
        # exit()


        # sparse_depth = depth.to_sparse()
        # indices = sparse_depth.indices()
        # values = sparse_depth.values()
        # xy_depth = torch.cat((indices.T, values[..., None]), dim=-1)
        # xy_depth = xy_depth.cuda()
        # xy_depth = xy_depth.unsqueeze(0)
        # xy_depth[..., -1] = xy_depth[..., -1] * 1000.
        # partial = cameras.unproject_points(xy_depth, True)
        # pc = PointCloud()
        # pc.points = Vector3dVector(partial[0].cpu().numpy())
        # o3d.visualization.draw_geometries([pc])
        # partial = partial.cpu()[0]
        #TODO END REMOVE

        partial = fast_from_depth_to_pointcloud(depth, cameras)

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

        # # TODO END REMOVE
        #
        # # Sample points from mesh
        # vertices = mesh.verts_packed()
        # complete = copy.deepcopy(vertices)  # TODO NEED TO ADD SOME NOISE
        #
        # keep = shuffle(list(range(len(vertices))))[:self.partial_points]
        # partial = copy.deepcopy(complete[keep])
        #
        # # Set partial_pcd such that it has the same size of the others
        # if partial.shape[0] < self.partial_points:
        #     diff = self.partial_points - partial.shape[0]
        #     partial = torch.cat((partial, torch.zeros(diff, 3)))
        #     padding_length = diff
        #
        # else:
        #     perm = torch.randperm(partial.size(0))
        #     ids = perm[:self.partial_points]
        #     partial = partial[ids]
        #
        # # Create implicit function input and output
        # imp_x, imp_y = sample_point_cloud(mesh,
        #                                   self.noise_rate,
        #                                   self.percentage_sampled,
        #                                   total=self.implicit_input_dimension,
        #                                   mode="unsigned")
        #
        # return label, image, partial, complete, imp_x, imp_y, padding_length

        # # TODO START OLD
        # tm = o3d.io.read_triangle_mesh(complete_path, False)
        # complete_pcd = tm.sample_points_uniformly(self.partial_points * self.multiplier_complete_sampling)
        # # TODO END
        #
        # # Get random position of camera
        # sph_radius = 1
        # y = random.uniform(-sph_radius, sph_radius)
        # theta = random.uniform(0, 2 * np.pi)
        # x = np.sqrt(sph_radius ** 2 - y ** 2) * cos(theta)
        # z = np.sqrt(sph_radius ** 2 - y ** 2) * sin(theta)
        # camera = [x, y, z]
        #
        # # Center to be in the middle
        # # points = np.array(complete_pcd.points)
        # # center = [max((points[:, 0] + min(points[:, 0]))/2),
        # #           max((points[:, 1] + min(points[:, 1]))/2),
        # #           max((points[:, 2] + min(points[:, 2]))/2)]
        # # center = np.array(center)[None, ...].repeat(len(points), axis=0)
        # # complete_pcd.points = Vector3dVector(points - center)
        #
        # # Remove hidden points
        # _, pt_map = complete_pcd.hidden_point_removal(camera, 500)  # radius * 4
        # partial_pcd = complete_pcd.select_by_index(pt_map)
        #
        # partial_pcd = torch.FloatTensor(np.array(partial_pcd.points))
        # # Set partial_pcd such that it has the same size of the others
        # if partial_pcd.shape[0] < self.partial_points:
        #     diff = self.partial_points - partial_pcd.shape[0]
        #     partial_pcd = torch.cat((partial_pcd, torch.zeros(diff, 3)))
        #     padding_length = diff
        #
        # else:
        #     perm = torch.randperm(partial_pcd.size(0))
        #     ids = perm[:self.partial_points]
        #     partial_pcd = partial_pcd[ids]
        #
        # if self.mode in ['valid', 'test']:
        #     if self.overfit_mode:
        #         mesh_path = TrainConfig.overfit_sample
        #     else:
        #         mesh_path = str(self.data_root / self.samples[idx].strip() / 'models/model_normalized.obj')
        #     return label, partial_pcd, mesh_path,
        #
        # complete_pcd = np.array(complete_pcd.points)
        # complete_pcd = torch.FloatTensor(complete_pcd)
        #
        # imp_x, imp_y = sample_point_cloud(tm,
        #                                   self.noise_rate,
        #                                   self.percentage_sampled,
        #                                   total=self.implicit_input_dimension,
        #                                   mode="unsigned")
        #
        # # imp_x, imp_y = andreas_sampling(tm, self.implicit_input_dimension)
        # imp_x, imp_y = torch.tensor(imp_x).float(), torch.tensor(imp_y).bool().float().bool().float()  # TODO oh god..
        #
        # return label, partial_pcd, complete_pcd, imp_x, imp_y, padding_length

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

        # lab, part, comp, x, y, pad = elem
        #
        # pc = PointCloud()
        # pc.points = Vector3dVector(comp)
        # o3d.visualization.draw_geometries([pc], window_name="Complete")
        # #
        # pc = PointCloud()
        # pc.points = Vector3dVector(part)
        # o3d.visualization.draw_geometries([pc], window_name="Partial")

        # print(lab)
        #
        # points = []
        # for _ in range(1000):
        #     points.append(np.array([1, random.uniform(-1, 1), random.uniform(-1, 1)]))
        #     points.append(np.array([-1, random.uniform(-1, 1), random.uniform(-1, 1)]))
        #     points.append(np.array([random.uniform(-1, 1), 1, random.uniform(-1, 1)]))
        #     points.append(np.array([random.uniform(-1, 1), -1, random.uniform(-1, 1)]))
        #     points.append(np.array([random.uniform(-1, 1), random.uniform(-1, 1), 1]))
        #     points.append(np.array([random.uniform(-1, 1), random.uniform(-1, 1), -1]))
        #
        # points = np.stack(points)
        # points = np.concatenate((points, comp))

        # pc = PointCloud()
        # pc.points = Vector3dVector(x)
        # colors = []
        # for i in y:
        #     if i == 0.:
        #         colors.append(np.array([1, 0, 0]))
        #     if i == 1.:
        #         colors.append(np.array([0, 1, 0]))
        # colors = np.stack(colors)
        # colors = Vector3dVector(colors)
        # pc.colors = colors
        # o3d.visualization.draw_geometries([pc])
