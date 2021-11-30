from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
from open3d.visualization import draw_geometries
from torch import cdist
from utils.misc import create_3d_grid
import open3d as o3d
from configs.server_config import ModelConfig, DataConfig
from datasets.ShapeNetPOVRemoval import BoxNet
from main import HyperNetwork
import torch
import numpy as np

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))

device = "cuda"


def fp_sampling(points, num, starting_point=None):
    batch_size = points.shape[0]
    # If no starting_point is provided, the starting point is the first point of points
    if starting_point is None:
        starting_point = points[:, 0].unsqueeze(1)
    D = cdist(starting_point, points).squeeze(1)

    perm = torch.zeros((batch_size, num), dtype=torch.int32, device=points.device)
    ds = D
    for i in range(0, num):
        idx = torch.argmax(ds, dim=1)
        perm[:, i] = idx
        ds = torch.minimum(ds, cdist(points[torch.arange(batch_size), idx].unsqueeze(1), points).squeeze())

    return perm


if __name__ == '__main__':
    from icub_demo_pc import create_rotation_matrix

    res = 0.01

    model = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
    model = model.to(device)
    model.eval()

    valid_set = BoxNet(DataConfig, 10000)

    for sample in valid_set:

        label, partial, mesh_raw, x, y = sample

        verts, tris = mesh_raw

        mesh = o3d.geometry.TriangleMesh(Vector3dVector(verts), Vector3iVector(tris))

        # TODO START REMOVE DEBUG
        partial_pc = PointCloud()
        partial_pc.points = Vector3dVector(np.array(partial))
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([partial_pc, coord])
        # TODO END REMOVE DEBUG

        partial = partial.unsqueeze(0)
        partial = partial.to(device)

        prediction = model(partial, step=res)

        # Get the selected point on the grid
        prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()
        samples = create_3d_grid(batch_size=partial.shape[0], step=res)  # TODO we create grid two times...
        samples = samples.squeeze(0).detach().cpu().numpy()
        selected = samples[prediction > 0.5]
        pred_pc = PointCloud()
        pred_pc.points = Vector3dVector(selected)
        pred_pc.estimate_normals()
        pred_pc.orient_normals_consistent_tangent_plane(10)

        # TODO REMOVE DEBUG (VISUALIZE PARTIAL POINT CLOUD AND PREDICTED POINT CLOUD)
        # partial = partial.squeeze(0).detach().cpu().numpy()  # TODO VISUALIZE DEBUG
        # part_pc = PointCloud()  # TODO VISUALIZE DEBUG
        # part_pc.points = Vector3dVector(partial)  # TODO VISUALIZE DEBUG
        # colors = np.array([0, 255, 0])[None, ...].repeat(partial.shape[0], axis=0)  # TODO VISUALIZE DEBUG
        # part_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        # colors = np.array([0, 0, 255])[None, ...].repeat(selected.shape[0], axis=0)  # TODO VISUALIZE DEBUG
        # pred_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        # o3d.visualization.draw_geometries([pred_pc, part_pc])  # TODO VISUALIZE DEBUG

        # Run RANSAC for every face
        centers = []
        aux_pc = PointCloud(pred_pc)
        for i in range(6):
            points = aux_pc.segment_plane(res, 10, 100)  # TODO FINE TUNE THIS PARAMETERS
            points_list = np.array(points[1])
            plane_points = np.array(aux_pc.points)[points_list]

            centers.append(np.mean(plane_points, axis=0))

            aux_pc = aux_pc.select_by_index(points[1], invert=True)
            o3d.visualization.draw_geometries([aux_pc])  # TODO VISUALIZE DEBUG

        # TODO VISUALIZE DEBUG (VISUALIZE RECONSTRUCTED MESH AND FACE CENTERS)
        # centers_pc = PointCloud()  # TODO VISUALIZE DEBUG
        # centers_pc.points = Vector3dVector(np.array(centers))  # TODO VISUALIZE DEBUG
        # colors = np.array([255, 0, 0])[None, ...].repeat(len(centers_pc.points), axis=0)  # TODO VISUALIZE DEBUG
        # centers_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        # o3d.visualization.draw_geometries([rec_mesh, centers_pc], mesh_show_back_face=True)  # TODO VISUALIZE DEBUG

        # Get closest point from sampled point cloud for every center
        true_centers = []
        sampled_mesh = torch.FloatTensor(np.array(pred_pc.points))
        centers = torch.FloatTensor(np.array(centers))
        for c in centers:
            c = c.unsqueeze(0)

            # TODO VISUALIZE DEBUG
            # c_pc = PointCloud()  # TODO VISUALIZE DEBUG
            # c_pc.points = Vector3dVector(np.array(c))  # TODO VISUALIZE DEBUG
            # colors = np.array([255, 0, 0])[None, ...].repeat(len(c), axis=0)  # TODO VISUALIZE DEBUG
            # c_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
            # o3d.visualization.draw_geometries([rec_pc, c_pc])  # TODO VISUALIZE DEBUG
            # TODO END VISUALIZE DEBUG

            dist = sampled_mesh - c
            dist = torch.square(dist[..., 0]) + torch.square(dist[..., 1]) + torch.square(dist[..., 2])
            true_centers.append(torch.argmin(dist).numpy())

        true_centers = np.array(true_centers).squeeze()

        # TODO VISUALIZE DEBUG (SHOWS RECONSTRUCTED MESH AND REAL CENTERS)
        final_points = pred_pc.select_by_index(true_centers)

        pred_pc.normals = Vector3dVector([])  # TODO VISUALIZE MEGA DEBUG

        geometries = [pred_pc, final_points]
        for c, normal in zip(np.array(final_points.points), np.array(final_points.normals)):
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=c, size=0.1)
            normal = normal/np.linalg.norm(normal)
            R = create_rotation_matrix(np.array([0, 0, 1]), normal)
            coord.rotate(R, center=c)
            geometries.append(coord)

        o3d.visualization.draw_geometries(geometries)
