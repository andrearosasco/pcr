import open3d as o3d
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from numpy.random import uniform
from open3d import visualization

try:
    from open3d.cuda.pybind import camera
    from open3d.cuda.pybind.geometry import PointCloud
    from open3d.cuda.pybind.utility import Vector3dVector
    from open3d.cuda.pybind.visualization import draw_geometries
except ImportError:
    from open3d.cpu.pybind import camera
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector
    from open3d.cpu.pybind.visualization import draw_geometries
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from utils.misc import sample_point_cloud


class BoxNet(data.Dataset):
    def __init__(self, config, n_samples):
        #  Backbone Input
        self.partial_points = config.partial_points

        # Implicit function input
        self.noise_rate = config.noise_rate
        self.implicit_input_dimension = config.implicit_input_dimension
        self.tolerance = config.tolerance
        self.dist = config.dist

        # Synthetic dataset
        self.n_samples = n_samples

        # Augmentation parameters
        self.cube_dim = {'low': 50, 'high': 600}
        self.camera_dist = {'low': 400, 'high': 1500}
        self.mesh_angle = {'low': 0, 'high': 360}

    def __getitem__(self, idx):  # Must return complete, imp_x and impl_y

        # Need to resample the image when the object is too far from the camera (the depth image is empty)
        #   or when, after normalization, the full point cloud doesn't fit the input/output space.
        while True:
            w, h, d = [uniform(**self.cube_dim) for _ in range(3)]
            mesh = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)
            mesh.translate(-mesh.get_center())

            p = uniform()
            if 0 <= p < .21:
                rotation = R.from_euler('z', uniform(**self.mesh_angle)).as_matrix()
            elif .21 <= p < .51:
                rotation = R.from_euler('xz', [uniform(**self.mesh_angle), uniform(**self.mesh_angle)]).as_matrix()
            elif .51 <= p < 1:
                rotation = R.random().as_matrix()

            mesh = mesh.rotate(rotation)

            # Define camera transformation and intrinsics
            #  (Camera is in the origin facing negative z, shifting it of z=1 puts it in front of the object)
            c_dist = uniform(**self.camera_dist)

            # TODO it looks like the camera takes the image from [0, 0, -dist] instead of [0, 0, dist]
            camera_parameters = camera.PinholeCameraParameters()
            camera_parameters.extrinsic = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, c_dist],
                                                    [0, 0, 0, 1]])

            # TODO for some reason we can't use the realsense parameter. The partial point cloud becomes not aligned.
            # intrinsics = {'fx': 384.025146484375, 'fy': 384.025146484375, 'cx': 319.09661865234375,
            #               'cy': 237.75723266601562,
            #               'width': 640, 'height': 480}

            intrinsics = {'fx': 1000, 'fy': 1000, 'cx': 959.5,
                          'cy': 539.5,
                          'width': 1920, 'height': 1080}

            camera_parameters.intrinsic.set_intrinsics(**intrinsics)

            # Move the view and take a depth image
            viewer = visualization.Visualizer()

            viewer.create_window(width=intrinsics['width'], height=intrinsics['height'], visible=False)
            viewer.clear_geometries()
            viewer.add_geometry(mesh)

            control = viewer.get_view_control()
            control.convert_from_pinhole_camera_parameters(camera_parameters)

            depth = viewer.capture_depth_float_buffer(True)

            viewer.remove_geometry(mesh)
            viewer.destroy_window()
            del control

            if np.array(depth).sum() == 0.0:
                continue

            # TODO the noise doesn't reflect the realsense stream noise
            depth = np.array(depth)
            sigma = 0.001063 + 0.0007278 * (depth*0.001) + 0.003949 * ((depth*0.001) ** 2)
            depth += (np.random.normal(0, 1, depth.shape) * (depth != 0) * sigma * 1000)

            # Normalize the partial point cloud (all we can do at test time)
            depth_image = o3d.geometry.Image(depth)

            # cv2.imshow('', np.array(depth))
            # cv2.waitKey(0)

            partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, camera_parameters.intrinsic, camera_parameters.extrinsic)
            partial_pcd = np.array(partial_pcd.points)

            # pc = PointCloud()
            # pc.points = Vector3dVector(partial_pcd)
            # draw_geometries([pc, mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=1000).transform([[1, 0, 0, 0],
            #                                             [0, 1, 0, 0],
            #                                             [0, 0, 1, dist],
            #                                             [0, 0, 0, 1]]),
            #                  o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=1000)])
            # print(dist)
            # print(np.mean(partial_pcd, axis=0))

            # Normalize the part ial point cloud (all we could do at test time)
            mean = np.mean(np.array(partial_pcd), axis=0)
            partial_pcd = np.array(partial_pcd) - mean
            var = np.sqrt(np.max(np.sum(partial_pcd ** 2, axis=1)))

            partial_pcd = partial_pcd / (var * 2) #(var * 2) (1040*2)

            # Move the mesh so that it matches the partial point cloud position
            # (the [0, 0, 1] is to compensate for the fact that the partial pc is in the camera frame)
            mesh.translate(-mean)
            mesh.scale(1 / (var * 2), center=[0, 0, 0]) # (var * 2) (1040*2)

            aux = np.array(mesh.vertices)
            t1 = np.all(np.min(aux, axis=0) > -0.5)
            t2 = np.all(np.max(aux, axis=0) < 0.5)

            if not (t1 and t2):
                # # # TODO Tests START
                # complete_pcd = mesh.sample_points_uniformly(self.partial_points)
                # complete_pcd.paint_uniform_color([1, 0, 0])
                #
                # pc = PointCloud()
                # pc.points = Vector3dVector(partial_pcd)
                # pc.paint_uniform_color([0, 0, 1])
                # draw_geometries([complete_pcd, create_cube(), pc])
                # # # TODO Tests END
                continue

            break


        # Sample labeled point on the mesh
        samples, occupancy = sample_point_cloud(mesh,
                                                n_points=self.implicit_input_dimension,
                                                dist=self.dist,
                                                noise_rate=self.noise_rate,
                                                tolerance=self.tolerance)

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
            diff = self.partial_points - partial_pcd.shape[0]
            partial_pcd = torch.cat((partial_pcd, torch.zeros(diff, 3)))

        samples = torch.tensor(samples).float()
        occupancy = torch.tensor(occupancy, dtype=torch.float)

        return 0, partial_pcd, [np.array(mesh.vertices), np.array(mesh.triangles)], samples, occupancy

    def __len__(self):
        return int(self.n_samples)


if __name__ == "__main__":
    from configs.local_config import DataConfig
    from tqdm import tqdm
    # from open3d.cpu.pybind.geometry import PointCloud
    # from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector

    a = DataConfig()
    a.dataset_path = Path("..", "data", "ShapeNetCore.v2")
    iterator = BoxNet(a, 10000)
    loader = DataLoader(iterator, num_workers=0, shuffle=False, batch_size=1)
    for elem in tqdm(loader):
        lab, part, mesh_vars, x, y = elem

        # verts, tris = mesh_vars
        #
        # mesh = o3d.geometry.TriangleMesh(Vector3dVector(verts[0].cpu()), Vector3iVector(tris[0].cpu()))
        # # o3d.visualization.draw_geometries([mesh], window_name="Complete")
        #
        # pc_part = PointCloud()
        # pc_part.points = Vector3dVector(part[0])  # remove batch dimension
        # # o3d.visualization.draw_geometries([pc_part], window_name="Partial")
        #
        # pc = PointCloud()
        # pc.points = Vector3dVector(x[0])  # remove batch dimension
        # colors = []
        # for i in y[0]:  # remove batch dimension
        #     if i == 0.:
        #         colors.append(np.array([1, 0, 0]))
        #     if i == 1.:
        #         colors.append(np.array([0, 1, 0]))
        # colors = np.stack(colors)
        # colors = Vector3dVector(colors)
        # pc.colors = colors
        # # o3d.visualization.draw_geometries([pc, mesh])
