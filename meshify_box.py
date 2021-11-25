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

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))

device = "cuda"


def fp_sampling(points, num):
    batch_size = points.shape[0]
    D = cdist(points[:, 0].unsqueeze(1), points).squeeze(1)
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = torch.zeros((batch_size, num), dtype=torch.int32, device=points.device)
    ds = D
    for i in range(1, num):
        idx = torch.argmax(ds, dim=1)
        perm[:, i] = idx
        ds = torch.minimum(ds, cdist(points[torch.arange(batch_size), idx].unsqueeze(1), points).squeeze())

    return perm


if __name__ == '__main__':

    model = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
    model = model.to(device)
    model.eval()

    valid_set = BoxNet(DataConfig, DataConfig.val_samples)

    for sample in valid_set:

        label, partial, mesh_raw, x, y = sample

        verts, tris = mesh_raw

        mesh = o3d.geometry.TriangleMesh(Vector3dVector(verts), Vector3iVector(tris))

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
        # draw_geometries([pred_pc, mesh])
        draw_geometries([pred_pc])

        # Create mesh from point cloud

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
