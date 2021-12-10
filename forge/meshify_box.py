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

    model = HyperNetwork.load_from_checkpoint('../checkpoint/best', config=ModelConfig)
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

        prediction = model(partial, step=0.01)

        # Visualize
        prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()

        samples = create_3d_grid(batch_size=partial.shape[0], step=0.01)  # TODO we create grid two times...
        samples = samples.squeeze(0).detach().cpu().numpy()

        selected = samples[prediction > 0.5]

        partial = partial.squeeze(0).detach().cpu().numpy()
        part_pc = PointCloud()
        part_pc.points = Vector3dVector(partial)

        pred_pc = PointCloud()
        pred_pc.points = Vector3dVector(selected)
        colors = np.array([0, 255, 0])[None, ...].repeat(selected.shape[0], axis=0)
        pred_pc.colors = Vector3dVector(colors)

        # Create mesh from point cloud
        hull, _ = pred_pc.compute_convex_hull()
        # o3d.visualization.draw_geometries([hull])

        convex_pc = PointCloud()
        convex_pc.points = hull.vertices

        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(convex_pc)
        for alpha in [0.5]:
            print(f"alpha={alpha:.3f}")
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                convex_pc, alpha, tetra_mesh, pt_map)
            rec_mesh.compute_vertex_normals()
            print("REC:", rec_mesh.is_watertight())
            print("ORIGINAL", mesh.is_watertight())
            o3d.visualization.draw_geometries([rec_mesh, mesh.translate([1, 1, 1]), part_pc.translate([1, 1, 1])], mesh_show_back_face=True)

        # TODO START NOT BAD, BUT WE HAVE ONLY 4 VERTICES
        # points_tensor = torch.FloatTensor(np.array(pred_pc.points)).unsqueeze(0)
        # first_vertex_id = fp_sampling(points_tensor, 1, starting_point=torch.FloatTensor([[0, 0, 0]]))
        # first_vertex_id = first_vertex_id[0][0].item()  # Get int from tensor
        # first_vertex = points_tensor[:, first_vertex_id, :]
        #
        # colors = np.array(pred_pc.colors)
        # colors[first_vertex_id] = np.array([255, 0, 0])
        # pred_pc.colors = Vector3dVector(colors)
        #
        # # o3d.visualization.draw_geometries([pred_pc])
        #
        # vertices = fp_sampling(points_tensor, 3, starting_point=first_vertex)
        # vertices = np.array(vertices[0])
        # colors = np.array(pred_pc.colors)
        # colors[vertices] = np.array([255, 0, 0])
        # pred_pc.colors = Vector3dVector(colors)
        #
        # o3d.visualization.draw_geometries([pred_pc])
        # TODO START NOT BAD, BUT WE HAVE ONLY 4 VERTICES

        # TODO NOT BAD, BUT STILL NOT WATERTIGHT
        # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pred_pc)
        # for alpha in [1, 0.04, 0.03, 0.02, 0.01]:
        #     print(f"alpha={alpha:.3f}")
        #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        #         pred_pc, alpha, tetra_mesh, pt_map)
        #     mesh.compute_vertex_normals()
        #     print(mesh.is_watertight())
        #     o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        # TODO END NOT BAD, BUT STILL NOT WATERTIGHT

        # TODO NOT BAD, WATERTIGHT, BUT INCONSISTENT
        # import pymeshfix
        # import pyvista as pv
        #
        # array = selected
        #
        # point_cloud = pv.PolyData(array)
        # surf = point_cloud.reconstruct_surface(nbr_sz=10, sample_spacing=0.01)
        #
        # mf = pymeshfix.MeshFix(surf)
        # mf.repair()
        # repaired = mf.mesh
        #
        # pl = pv.Plotter()
        # pl.add_mesh(point_cloud, color='k', point_size=10)
        # pl.add_mesh(repaired)
        # pl.add_title('Reconstructed Surface')
        # pl.show()
        # TODO NOT BAD, WATERTIGHT, BUT INCONSISTENT
