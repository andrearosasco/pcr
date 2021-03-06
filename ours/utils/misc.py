import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc
import tqdm
try:
    from open3d.cuda.pybind.geometry import PointCloud
    from open3d.cuda.pybind.utility import Vector3dVector
except (ModuleNotFoundError, ImportError):
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector
#from pytorch3d.ops import sample_points_from_meshes
#from pytorch3d.loss.point_mesh_distance import point_face_distance
#from pytorch3d.renderer import look_at_view_transform, Materials, PointLights, BlendParams
import random
#from pytorch3d.renderer import PerspectiveCameras
#from pytorch3d.io import load_objs_as_meshes
#from pytorch3d.structures import Meshes
#from pytorch3d.renderer import RasterizationSettings
from cmath import cos
from cmath import sin
#from pytorch3d.renderer import SoftPhongShader
#from pytorch3d.renderer import MeshRasterizer
from utils.fps import fp_sampling
from math import ceil, cos, sin
import open3d as o3d


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = fp_sampling(data, number)
    fps_data = data[torch.arange(fps_idx.shape[0]).unsqueeze(-1), fps_idx.long(), :]

    return fps_data


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_lambda_sche(opti, config):
    if config.get('decay_step') is not None:
        lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler


def build_lambda_bnsche(model, config):
    if config.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.bn_momentum * config.bn_decay ** (e / config.decay_step), config.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def chamfer(samples, predictions, meshes):
    mesh_paths, rotations, means, vars = meshes
    for p, r, m, v, pred in zip(mesh_paths, rotations, means, vars, predictions):
        query = samples[0, (pred > 0.5).squeeze()]

        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.io.read_triangle_mesh(p, False)
        mesh.rotate(r.cpu().numpy())
        mesh.translate(-m.cpu().numpy())
        mesh.scale(1 / (v.cpu().numpy() * 2), center=[0, 0, 0])
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh)
        query_points = o3d.core.Tensor(query.cpu().numpy(), dtype=o3d.core.Dtype.Float32)
        signed_distance = scene.compute_distance(query_points)
        return np.mean(signed_distance.numpy())

def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


def seprate_point_cloud(xyz, num_points, crop, fixed_points=None, padding_zeros=False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).to(xyz.device)
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).to(xyz.device)

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p=2, dim=-1)  # 1 1 2048

        idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]  # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0)  # 1 N 3

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):
            INPUT.append(fps(input_data, 2048))
            CROP.append(fps(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)  # B N 3
    crop_data = torch.cat(CROP, dim=0)  # B M 3

    return input_data.contiguous(), crop_data.contiguous()


def pc_grid_reconstruction(model, min_value=-1, max_value=1, step=0.05, just_true=False, bs=2, nz=41,
                           device="cuda:0"):
    """
    # GET AN IMPLICIT FUNCTION THAT WORKS WITH BATCHES AND DOES UNIFORM SAMPLING FOR bs BATCHES
    :param model: Iterable of ImplicitFunction
    :param min_value:
    :param max_value:
    :param step:
    :param just_true:
    :param bs: expected batch size
    :param nz: number of layer to give in a single time
    :param device: device where to do operation
    :return: FloatTensor( n_points, 3 if just_true else 4 ( also class ) )
    """
    x_range = torch.FloatTensor(np.arange(min_value, max_value + step, step)).to(device)
    y_range = torch.FloatTensor(np.arange(min_value, max_value + step, step)).to(device)
    z_range = torch.FloatTensor(np.arange(min_value, max_value + step, step)).to(device)
    grid_2d = torch.cartesian_prod(x_range, y_range)

    n_cycle = range(ceil(len(z_range) / nz))

    results = []
    for i in tqdm.tqdm(n_cycle):
        useful = []
        for k in range(nz):
            if i * nz + k < len(z_range):
                useful.append(z_range[i * nz + k])

        actual = len(useful)
        zs = torch.stack(useful).unsqueeze(-1).T.repeat(len(grid_2d), 1).T.reshape((len(grid_2d) * actual, 1))
        grid_3d = torch.cat((grid_2d.repeat(actual, 1), zs), dim=1)

        # TODO REMOVE DEBUG
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(grid_3d)
        # draw_geometries([pcd])
        # TODO END DEBUG

        grid_3d = grid_3d.repeat(bs, 1, 1)  # add batch dimension
        res = model(grid_3d)  # remove batch dimension
        grid_3d = grid_3d.squeeze()

        result = torch.cat((grid_3d, res), dim=-1)
        if just_true:
            good_ids = torch.nonzero(result[..., -1] == 1.).squeeze(1)
            result = result[good_ids]
            results.append(result[..., :-1])
        else:
            results.append(result)
    results = torch.cat([t for t in results], dim=1)
    return results


def sample_point_cloud_pytorch3d(mesh, noise_rate=0.1, percentage_sampled=0.1, total=8192, tollerance=0.0001):
    n_points_uniform = int(total * percentage_sampled)
    n_points_surface = total - n_points_uniform

    points_uniform = torch.rand(1, n_points_uniform, 3, device=mesh.device) * 2 - 1

    points_surface = sample_points_from_meshes(mesh, n_points_surface)

    points_surface = points_surface + (noise_rate * torch.randn_like(points_surface).to(mesh.device))

    points = torch.cat((points_uniform, points_surface), axis=1)

    points_first_idx = torch.LongTensor([0]).to(mesh.device)
    tris = mesh.verts_packed()[mesh.faces_packed()]

    distances = point_face_distance(points.squeeze(), points_first_idx, tris, points_first_idx, points.shape[1])

    distances = distances < tollerance

    return points, distances


def create_cube():
    cube = []
    for _ in range(2500):
        p = np.random.rand((3)) - 0.5
        cube.append([-0.5, p[1], p[2]])
        cube.append([0.5, p[1], p[2]])
        cube.append([p[0], -0.5, p[2]])
        cube.append([p[0], 0.5, p[2]])
        cube.append([p[0], p[1], -0.5])
        cube.append([p[0], p[1], 0.5])

    cb = PointCloud()
    cb.points = Vector3dVector(np.array(cube))
    cb.paint_uniform_color([0, 1, 0])

    return cb


def create_sphere():
    sphere = []
    for _ in range(10000):
        sph_radius = 0.5
        y = random.uniform(-sph_radius, sph_radius)
        theta = random.uniform(0, 2 * np.pi)
        x = np.sqrt(sph_radius ** 2 - y ** 2) * cos(theta)
        z = np.sqrt(sph_radius ** 2 - y ** 2) * sin(theta)
        sphere.append([x, y, z])

    sph = PointCloud()
    sph.points = Vector3dVector(np.array(sphere))
    sph.paint_uniform_color([1, 0, 0])

    return sph


def sample_point_cloud(mesh, noise_rate=0.1, percentage_sampled=0.1, total=8192, tollerance=0.01, mode="unsigned"):
    """
    http://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html
    Produces input for implicit function
    :param mesh: Open3D mesh
    :param noise_rate: rate of gaussian noise added to the point sampled from the mesh
    :param percentage_sampled: percentage of point that must be sampled uniform
    :param total: total number of points that must be returned
    :param tollerance: maximum distance from mesh for a point to be considered 1.
    :param mode: str, one in ["unsigned", "signed", "occupancy"]
    :return: points (N, 3), occupancies (N,)
    """
    # TODO try also with https://blender.stackexchange.com/questions/31693/how-to-find-if-a-point-is-inside-a-mesh
    n_points_uniform = int(total * percentage_sampled)
    n_points_surface = total - n_points_uniform

    points_uniform = np.random.rand(n_points_uniform, 3) - 0.5

    points_surface = np.array(mesh.sample_points_uniformly(n_points_surface).points)

    # TODO REMOVE DEBUG ( VISUALIZE POINT CLOUD SAMPLED FROM THE SURFACE )
    # from open3d.open3d.geometry import PointCloud
    # pc = PointCloud()
    # pc.points = Vector3dVector(points_surface)
    # open3d.visualization.draw_geometries([pc])

    points_surface = points_surface + (noise_rate * np.random.randn(len(points_surface), 3))

    # TODO REMOVE DEBUG ( VISUALIZE POINT CLOUD FROM SURFACE + SOME NOISE )
    # pc = PointCloud()
    # pc.points = Vector3dVector(points_surface)
    # open3d.visualization.draw_geometries([pc])

    points = np.concatenate([points_uniform, points_surface], axis=0)

    # TODO REMOVE DEBUG ( VISUALIZE ALL POINTS WITHOUT LABEL )
    # pc = PointCloud()
    # pc.points = Vector3dVector(points)
    # open3d.visualization.draw_geometries([pc])

    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh)
    query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)

    if mode == "unsigned":
        unsigned_distance = scene.compute_distance(query_points)
        occupancies1 = -tollerance < unsigned_distance
        occupancies2 = unsigned_distance < tollerance
        occupancies = occupancies1 & occupancies2
    elif mode == "signed":
        signed_distance = scene.compute_signed_distance(query_points)
        occupancies = signed_distance < tollerance  # TODO remove this to deal with distances
    elif mode == "occupancies":
        occupancies = scene.compute_occupancy(query_points)
    else:
        raise NotImplementedError("Mode not implemented")

    return points, occupancies.numpy()


def get_ptcloud_img(ptcloud):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(30, 45)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def visualize_KITTI(path, data_list, titles=['input', 'pred'], cmap=['bwr', 'autumn'], zdir='y',
                    xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)):
    fig = plt.figure(figsize=(6 * len(data_list), 6))
    cmax = data_list[-1][:, 0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:, 0] / cmax
        ax = fig.add_subplot(1, len(data_list), i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=color, vmin=-1, vmax=1, cmap=cmap[0], s=4,
                       linewidth=0.05, edgecolors='black')
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)

    pic_path = path + '.png'
    fig.savefig(pic_path)

    np.save(os.path.join(path, 'input.npy'), data_list[0].numpy())
    np.save(os.path.join(path, 'pred.npy'), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e // 50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1, 1))[0, 0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim=1)
    return pc


#
# def random_scale(partial, scale_range=[0.8, 1.2]):
#     scale = torch.rand(1).to(xyz.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
#     return partial * scale


def create_3d_grid(min_value=-0.5, max_value=0.5, step=0.04, batch_size=1):
    x_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    y_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    z_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    grid_2d = torch.cartesian_prod(x_range, y_range)
    grid_2d = grid_2d.repeat(x_range.shape[0], 1)
    z_repeated = z_range.unsqueeze(1).T.repeat(x_range.shape[0] ** 2, 1).T.reshape(-1)[..., None]
    grid_3d = torch.cat((grid_2d, z_repeated), dim=-1)
    grid_3d = grid_3d.unsqueeze(0).repeat(batch_size, 1, 1)
    return grid_3d


def check_mesh_contains(meshes, queries, max_dist=0.01):
    occupancies = []
    queries = queries.detach().cpu().numpy()
    for mesh, query in zip(meshes, queries):
        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh)
        query_points = o3d.core.Tensor(query, dtype=o3d.core.Dtype.Float32)
        signed_distance = scene.compute_distance(query_points)
        occupancies.append((signed_distance < max_dist).numpy())
    occupancies = np.stack(occupancies)[..., None]
    return occupancies.astype(float)


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


def fast_from_depth_to_pointcloud(depth, cameras, R, T):

    cx = 312.9869
    cy = 241.3109
    fy = 1067.487
    fx = 1066.778

    cx = 114.8
    cy = 31.75
    fy = 76.2
    fx = 76.2

    k = torch.eye(4).to(depth.device)
    k[0, :3] = torch.FloatTensor([1 / fx, 0, -(cx * fy) / (fx * fy)]).to(depth.device)
    k[1, 1:3] = torch.FloatTensor([1 / fy, -cy / fy]).to(depth.device)

    sparse_depth = depth.to_sparse()
    indices = sparse_depth.indices()
    values = sparse_depth.values()
    xy_depth = torch.cat((indices.T, values[..., None]), dim=-1)

    import copy
    final_z = copy.deepcopy(xy_depth[..., -1])

    xy_depth = torch.cat((xy_depth, 1 / xy_depth[..., -1].unsqueeze(-1)), dim=-1)
    xy_depth[:, 2] = 1.
    points = xy_depth @ k.T * final_z.unsqueeze(-1)
    points = points[:, 0:3]

    R = R.cuda()
    T = T.cuda()

    points = points - T[0]
    points = points @ R[0]

    # TODO normalize points
    # center_x = (max(points[:, 0]) + min(points[:, 0])) / 2
    # center_y = (max(points[:, 1]) + min(points[:, 1])) / 2
    # center_z = (max(points[:, 2]) + min(points[:, 2])) / 2
    # center = torch.FloatTensor([center_x, center_y, center_z]).cuda()
    # points = points - center
    #
    # dist = max(torch.sqrt(torch.square(points[:, 0]) + torch.square(points[:, 1]) + torch.square(points[:, 2])))
    # points = points * (1.0 / float(dist))  # scale

    # TODO START DEBUG
    # for i in range(1000):
    #     sph_radius = 1
    #     y_light = random.uniform(-sph_radius, sph_radius)
    #     theta = random.uniform(0, 2 * np.pi)
    #     x_light = np.sqrt(sph_radius ** 2 - y_light ** 2) * cos(theta)
    #     z_light = np.sqrt(sph_radius ** 2 - y_light ** 2) * sin(theta)
    #     points = torch.cat((points, torch.FloatTensor([x_light, y_light, z_light]).unsqueeze(0).to(depth.device)))
    #
    # coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
   # TODO END DEBUG
    return points


def from_depth_to_pointcloud(depth, cameras):
    K = cameras.get_projection_transform().get_matrix()
    K = K.squeeze().cuda()

    fx = K[0, 0]
    cx = K[0, 2]
    fy = K[1, 1]
    cy = K[1, 2]
    #
    cx = 312.9869
    cy = 241.3109
    fy = 1067.487
    fx = 1066.778

    k = np.eye(4)
    k[0, :3] = np.array([1 / fx, 0, -(cx * fy) / (fx * fy)])
    k[1, 1:3] = np.array([1 / fy, -cy / fy])
    z_scale = 10000.0

    # depth = depth / z_scale

    points = []
    for u in range(depth.shape[0]):
        for v in range(depth.shape[1]):
            z = depth[u, v]
            if z != 0:
                xyz = (z * k @ np.array([u, v, 1, 1 / z]))[0:3]
                points.append(xyz)

    return np.stack(points)


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs):
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


def get_mesh_image(path=None, rotation=None, width=0, height=0, fx=0, fy=0, cx=0, cy=0, dist=0, device=None):
    # Load mesh with pytorch3d
    device = torch.device(device)  # TODO FIX
    mesh = load_objs_as_meshes([path], load_textures=True, create_texture_atlas=True, device=device)

    # Rotate it like before
    a = (mesh.verts_packed().cpu() @ rotation.T)
    b = mesh.faces_packed().cpu()
    mesh = mesh.cpu()
    c = mesh.textures
    mesh = Meshes([a.float()], [b], c)

    # points = mesh.verts_packed()  # TODO REMOVE
    # pc = PointCloud()  # TODO REMOVE
    # pc.points = Vector3dVector(points.cpu())  # TODO REMOVE
    # o3d.visualization.draw_geometries([pc])  # TODO REMOVE

    mesh = mesh.cuda()

    # Set camera
    R = np.array([[[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]])
    T = np.array([[0, 0, 1]])
    focal_length = torch.FloatTensor([fx, fy]).unsqueeze(0)
    principal_points = torch.FloatTensor([cx, cy]).unsqueeze(0)
    camera = PerspectiveCameras(focal_length=focal_length, principal_point=principal_points,
                                T=T, R=R, device=device, in_ndc=False)

    # Init rasterizer settings
    raster_settings = RasterizationSettings(
        image_size=(height, width), blur_radius=0.0, faces_per_pixel=5
    )

    # Init shader settings
    materials = Materials(device=device)

    # Light
    sph_radius = random.uniform(3, 4)
    y_light = random.uniform(-sph_radius, sph_radius)
    theta = random.uniform(0, 2 * np.pi)
    x_light = np.sqrt(sph_radius ** 2 - y_light ** 2) * cos(theta)
    z_light = np.sqrt(sph_radius ** 2 - y_light ** 2) * sin(theta)
    lights = PointLights(location=[[x_light, y_light, z_light]], device=device)
    blend_params = BlendParams(
        sigma=1e-1,
        gamma=1e-4,
        background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
    )

    # Init renderer
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=SoftPhongShader(
            lights=lights,
            cameras=camera,
            materials=materials,
            blend_params=blend_params,
        ),
    )

    # Render and return
    images, _ = renderer(mesh, image_size=(height, width))  # cull_backface=True
    return images[0, ..., :3].cpu().numpy()

# if __name__ == "__main__":
#     grid = create_3d_grid()
#     pc = PointCloud()
#     pc.points = Vector3dVector(grid.squeeze())
#     o3d.visualization.draw_geometries([pc], window_name=str(len(grid.squeeze())))
