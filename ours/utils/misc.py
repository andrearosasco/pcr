import numpy as np
import matplotlib.pyplot as plt
import open3d
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc

from torch import randn_like

from utils.fps import fp_sampling


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



def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).to(xyz.device)
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).to(xyz.device)

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()

def pc_grid_reconstruction(model, min_value=-1, max_value=1, step=0.05):
    x_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    y_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    z_range = torch.FloatTensor(np.arange(min_value, max_value + step, step))
    grid_2d = torch.cartesian_prod(x_range, y_range)
    results = []
    for z in tqdm.tqdm(z_range):
        repeated_z = z.reshape(1, 1).repeat(grid_2d.shape[0], 1)
        grid_3d = torch.cat((grid_2d, repeated_z), dim=1)

        # TODO REMOVE DEBUG
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(grid_3d)
        # draw_geometries([pcd])
        # TODO END DEBUG

        result = model(grid_3d)
        result = torch.cat((grid_3d, result), dim=-1)
        good_ids = torch.nonzero(result[..., -1] == 1.).squeeze(1)
        result = result[good_ids]
        results.append(result[..., :-1])

    return torch.cat(results, dim=0)


# Test function
if __name__ == "__main__":

    def model(elem):
        import random
        if random.random() > 0.5:
            return torch.zeros_like(elem).fill_(1.)[:, :1]
        else:
            return torch.zeros_like(elem).fill_(0.)[:, :1]

    res = pc_grid_reconstruction(model)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(res)
    draw_geometries([pcd])


def sample_point_cloud(xyz, voxel_size=0.1, noise_rate=0.1, percentage_sampled=0.1):  # 1 2048 3
    # TODO try also with https://blender.stackexchange.com/questions/31693/how-to-find-if-a-point-is-inside-a-mesh
    # VOXEL
    xyz = xyz.cpu()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    voxel_grid = open3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    # open3d.open3d.visualization.draw_geometries([voxel_grid])

    # ORIGINAL PC
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(xyz)
    # open3d.open3d.visualization.draw_geometries([pcd])

    # WITH GAUSSIAN NOISE
    z = randn_like(xyz) * noise_rate
    xyz = xyz + z
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    # open3d.open3d.visualization.draw_geometries([pcd])

    # WITH 10% UNIFORM RANDOM POINTS
    k = int(len(xyz) * percentage_sampled)
    random_points = torch.FloatTensor(k, 3).uniform_(-1, 1)
    pcd = open3d.geometry.PointCloud()
    xyz = torch.cat((xyz, random_points))
    pcd.points = open3d.utility.Vector3dVector(xyz)
    # open3d.open3d.visualization.draw_geometries([pcd])

    # WITH LABEL
    results = voxel_grid.check_if_included(open3d.utility.Vector3dVector(xyz))
    colors = np.zeros((len(xyz), 3))
    for i, value in enumerate(results):
        if value:
            colors[i] = np.array([0, 0, 1])
        else:
            colors[i] = np.array([0, 1, 0])
    pcd.colors = open3d.utility.Vector3dVector(colors)
    # open3d.open3d.visualization.draw_geometries([pcd])

    # t = 0
    # f = 0
    # for elem in results:
    #     if elem:
    #         t += 1
    #     else:
    #         f += 1
    # print("Found ", t, " points inside the voxels and ", f, " points outside the voxel")

    return np.array(pcd.points), np.array(results)


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
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img



def visualize_KITTI(path, data_list, titles = ['input','pred'], cmap=['bwr','autumn'], zdir='y', 
                         xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1) ):
    fig = plt.figure(figsize=(6*len(data_list),6))
    cmax = data_list[-1][:,0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:,0] /cmax
        ax = fig.add_subplot(1, len(data_list) , i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=color,vmin=-1,vmax=1 ,cmap = cmap[0],s=4,linewidth=0.05, edgecolors = 'black')
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
    up_num = max(64, 768 // (e//50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1,1))[0,0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim = 1)
    return pc
    
#
# def random_scale(partial, scale_range=[0.8, 1.2]):
#     scale = torch.rand(1).to(xyz.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
#     return partial * scale
