from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
import random
import os
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))


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
    import open3d as o3d
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


def gen_box(min_side=0.05, max_side=0.4):
    import open3d as o3d
    sizes = []
    for i in range(3):
        sizes.append(random.uniform(min_side, max_side))
    cube_mesh = o3d.geometry.TriangleMesh.create_box(sizes[0], sizes[1], sizes[2])

    colors = []
    for i in range(3):
        colors.append(random.uniform(0.0, 1.0))
    cube_mesh.paint_uniform_color(colors)
    cube_mesh.compute_vertex_normals()

    return cube_mesh


class BoxNet(data.Dataset):
    def __init__(self, config, mode="easy/train", overfit_mode=False):
        self.mode = mode
        self.overfit_mode = overfit_mode
        #  Backbone Input
        self.data_root = Path(config.dataset_path)
        self.partial_points = config.partial_points
        self.multiplier_complete_sampling = config.multiplier_complete_sampling

        # Implicit function input
        self.noise_rate = config.noise_rate
        self.percentage_sampled = config.percentage_sampled
        self.implicit_input_dimension = config.implicit_input_dimension

        # Synthetic dataset
        self.output_path = ".." + os.sep + "synthetic_boxes"
        self.n_samples = config.n_samples

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y

        # Find the mesh
        mesh = gen_box()
        complete_path = self.output_path + os.sep + "box_{}.obj".format(idx)

        while True:
            rotation = R.random().as_matrix()
            mesh = mesh.rotate(rotation)

            # Define camera transformation and intrinsics
            #  (Camera is in the origin facing negative z, shifting it of z=1 puts it in front of the object)
            dist = 1.5

            complete_pcd = mesh.sample_points_uniformly(self.partial_points * self.multiplier_complete_sampling)
            _, pt_map = complete_pcd.hidden_point_removal([0, 0, dist], 1000)  # radius * 4
            partial_pcd = complete_pcd.select_by_index(pt_map)

            if len(np.array(partial_pcd.points)) != 0:
                break

        # Normalize the partial point cloud (all we could do at test time)
        partial_pcd = np.array(partial_pcd.points)
        mean = np.mean(np.array(partial_pcd), axis=0)
        partial_pcd = np.array(partial_pcd) - mean
        var = np.sqrt(np.max(np.sum(partial_pcd ** 2, axis=1)))

        partial_pcd = partial_pcd / (var * 2)

        # Move the mesh so that it matches the partial point cloud position
        # (the [0, 0, 1] is to compensate for the fact that the partial pc is in the camera frame)
        mesh.translate(-mean)
        mesh.scale(1 / (var * 2), center=[0, 0, 0])

        # Sample labeled point on the mesh
        samples, occupancy = sample_point_cloud(mesh,
                                                self.noise_rate,
                                                self.percentage_sampled,
                                                total=self.implicit_input_dimension,
                                                mode="unsigned")

        # Next lines bring the shape a face of the cube so that there's more space to
        # complete it. But is it okay for the input to be shifted toward -0.5 and not
        # centered on the origin?
        #
        # normalized[..., 2] = normalized[..., 2] + (-0.5 - min(normalized[..., 2]))

        partial_pcd = torch.FloatTensor(partial_pcd)

        # Set partial_pcd such that it has the same size of the others
        if partial_pcd.shape[0] > self.partial_points:
            perm = torch.randperm(partial_pcd.size(0))
            ids = perm[:self.partial_points]
            partial_pcd = partial_pcd[ids]
        else:
            print(f'Warning: had to pad the partial pcd {complete_path} - points {partial_pcd.shape[0]} added {self.partial_points - partial_pcd.shape[0]}')
            diff = self.partial_points - partial_pcd.shape[0]
            partial_pcd = torch.cat((partial_pcd, torch.zeros(diff, 3)))

        samples = torch.tensor(samples).float()
        occupancy = torch.tensor(occupancy, dtype=torch.float) / 255

        return 0, partial_pcd, [np.array(mesh.vertices), np.array(mesh.triangles)], samples, occupancy

    def __len__(self):
        return int(self.n_samples)


if __name__ == "__main__":
    from configs.local_config import DataConfig
    from tqdm import tqdm
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    import open3d as o3d

    a = DataConfig()
    a.dataset_path = Path("..", "data", "ShapeNetCore.v2")
    iterator = BoxNet(a)
    loader = DataLoader(iterator, num_workers=0, shuffle=False, batch_size=1)
    for elem in tqdm(loader):
        lab, part, mesh_vars, x, y = elem

        verts, tris = mesh_vars

        mesh = o3d.geometry.TriangleMesh(Vector3dVector(verts[0].cpu()), Vector3iVector(tris[0].cpu()))
        # o3d.visualization.draw_geometries([mesh], window_name="Complete")

        pc_part = PointCloud()
        pc_part.points = Vector3dVector(part[0])  # remove batch dimension
        # o3d.visualization.draw_geometries([pc], window_name="Partial")

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 1.5])
        o3d.visualization.draw_geometries([pc_part, mesh, coord], window_name="Both")

        pc = PointCloud()
        pc.points = Vector3dVector(x[0])  # remove batch dimension
        colors = []
        for i in y[0]:  # remove batch dimension
            if i == 0.:
                colors.append(np.array([1, 0, 0]))
            if i == 1.:
                colors.append(np.array([0, 1, 0]))
        colors = np.stack(colors)
        colors = Vector3dVector(colors)
        pc.colors = colors
        o3d.visualization.draw_geometries([pc, mesh])
