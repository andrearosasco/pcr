##########################################################
# NOT BAD, but the mesh has alternating triangle normals #
##########################################################
import time
import yarp
import cv2
import numpy as np
import open3d as o3d
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from main import HyperNetwork
from configs.server_config import ModelConfig
from torch import cdist
from utils.misc import create_3d_grid

device = "cuda"


# TODO put presentation on teams, do video where I personally test model trained on AMI


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


class iCubGazebo:

    def __init__(self, rgb_port="/icubSim/cam/left/rgbImage:o", depth_port='/icubSim/cam/left/depthImage:o'):
        yarp.Network.init()

        # Create a port and connect it to the iCub simulator virtual camera
        self.rgb_port, self.depth_port = yarp.Port(), yarp.Port()
        self.rgb_port.open("/rgb-port")
        self.depth_port.open("/depth-port")
        yarp.Network.connect(rgb_port, "/rgb-port")
        yarp.Network.connect(depth_port, "/depth-port")

        self.rgb_array = np.zeros((240, 320, 3), dtype=np.uint8)
        self.rgb_image = yarp.ImageRgb()
        self.rgb_image.resize(320, 240)
        self.rgb_image.setExternal(self.rgb_array, self.rgb_array.shape[1], self.rgb_array.shape[0])

        self.depth_array = np.zeros((240, 320), dtype=np.float32)
        self.depth_image = yarp.ImageFloat()
        self.depth_image.resize(320, 240)
        self.depth_image.setExternal(self.depth_array, self.depth_array.shape[1], self.depth_array.shape[0])

    def read(self):
        self.rgb_port.read(self.rgb_image)
        self.depth_port.read(self.depth_image)

        return self.rgb_array[..., ::-1], self.depth_array


if __name__ == '__main__':

    alpha = 0.5
    res = 0.01

    # Setting up environment
    icub = iCubGazebo()
    model = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
    model = model.to(device)
    model.eval()

    # Main loop
    while True:

        # Get image
        rgb, depth = icub.read()

        # Get only red part
        rgb_mask = rgb[..., 2] == 102  # Red is the last dimension
        rgb_mask = rgb_mask.astype(float) * 255

        # Get only depth of the box
        filtered_depth = np.where(rgb_mask, depth, 0.)
        filtered_depth_img = filtered_depth.astype(float) * 255

        # TODO REMOVE DEBUG (VISUALIZE ICUB EYES OUTPUT)
        # cv2.imshow('RGB', rgb)  # TODO VISUALIZE DEBUG
        # cv2.imshow('Depth', depth)  # TODO VISUALIZE DEBUG
        # cv2.imshow('Mask', rgb_mask)  # TODO VISUALIZE DEBUG
        # cv2.imshow('Filtered Depth', filtered_depth_img)  # TODO VISUALIZE DEBUG

        # Convert depth image to Point Cloud
        fx = fy = 343.12110728152936
        cx = 160.0
        cy = 120.0
        intrinsics = o3d.camera.PinholeCameraIntrinsic(320, 240, fx, fy, cx, cy)
        partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(filtered_depth), intrinsics)
        # o3d.visualization.draw_geometries([pc])  # TODO VISUALIZE DEBUG

        # Sample Point Cloud
        partial = torch.FloatTensor(np.array(partial_pcd.points))  # Must be 1, 2024, 3
        start = time.time()  # TODO REMOVE DEBUG
        indices = fp_sampling(partial.unsqueeze(0), 2024)  # TODO THIS IS SLOW
        print("fps took: {}".format(time.time() - start))  # TODO REMOVE DEBUG
        partial = partial[indices.long().squeeze()]

        # Normalize Point Cloud as training time
        partial = np.array(partial)
        mean = np.mean(np.array(partial), axis=0)
        partial = np.array(partial) - mean
        var = np.sqrt(np.max(np.sum(partial ** 2, axis=1)))
        partial = partial / (var * 2)

        partial[..., -1] = -partial[..., -1]  # TODO VERIFY (IS IT NORMAL THAT I NEED TO INVERT THIS?)

        # TODO REMOVE DEBUG (VISUALIZE NORMALIZED PARTIAL POINT CLOUD)
        # partial_pcd = PointCloud()  # TODO VISUALIZE DEBUG
        # partial_pcd.points = Vector3dVector(partial)  # TODO VISUALIZE DEBUG
        # colors = np.array([0, 255, 0])[None, ...].repeat(partial.shape[0], axis=0)  # TODO VISUALIZE DEBUG
        # partial_pcd.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        # coord = o3d.geometry.TriangleMesh.create_coordinate_frame()  # TODO VISUALIZE DEBUG
        # o3d.visualization.draw_geometries([partial_pcd])  # TODO VISUALIZE DEBUG

        # Inference
        partial = torch.FloatTensor(partial).unsqueeze(0).to(device)
        prediction = model(partial, step=res)  # TODO step SHOULD BE 0.01

        # Get the selected point on the grid
        prediction = prediction.squeeze(0).squeeze(-1).detach().cpu().numpy()
        samples = create_3d_grid(batch_size=partial.shape[0], step=res)  # TODO we create grid two times...
        samples = samples.squeeze(0).detach().cpu().numpy()
        selected = samples[prediction > 0.5]
        pred_pc = PointCloud()
        pred_pc.points = Vector3dVector(selected)

        # TODO REMOVE DEBUG (VISUALIZE PARTIAL POINT CLOUD AND PREDICTED POINT CLOUD)
        # partial = partial.squeeze(0).detach().cpu().numpy()  # TODO VISUALIZE DEBUG
        # part_pc = PointCloud()  # TODO VISUALIZE DEBUG
        # part_pc.points = Vector3dVector(partial)  # TODO VISUALIZE DEBUG
        # colors = np.array([0, 255, 0])[None, ...].repeat(partial.shape[0], axis=0)  # TODO VISUALIZE DEBUG
        # part_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        #
        # colors = np.array([0, 0, 255])[None, ...].repeat(selected.shape[0], axis=0)  # TODO VISUALIZE DEBUG
        # pred_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        #
        # o3d.visualization.draw_geometries([pred_pc, part_pc])  # TODO VISUALIZE DEBUG

        # Create mesh from point cloud
        hull, _ = pred_pc.compute_convex_hull()
        # o3d.visualization.draw_geometries([hull])  # TODO VISUALIZE DEBUG (VISUALIZE CONVEX HULL)

        convex_pc = PointCloud()
        convex_pc.points = hull.vertices

        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(convex_pc)
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            convex_pc, alpha, tetra_mesh, pt_map)
        # rec_mesh.compute_vertex_normals()
        print("REC:", rec_mesh.is_watertight())
        o3d.visualization.draw_geometries([rec_mesh])  # TODO VISUALIZE DEBUG (SHOWS RECONSTRUCTED MESH)

        # Uniform sampling the reconstructed mesh
        rec_pc = rec_mesh.sample_points_uniformly(number_of_points=10000)
        colors = np.array([0, 0, 255])[None, ...].repeat(len(rec_pc.points), axis=0)
        rec_pc.colors = Vector3dVector(colors)
        # o3d.visualization.draw_geometries([rec_pc])  # TODO VISUALIZE DEBUG (POINT CLOUD FROM RECONSTRUCTED MESH)

        # Run RANSAC for every face
        centers = []
        aux_pc = PointCloud(rec_pc)
        for i in range(6):
            points = aux_pc.segment_plane(res, 10, 100)  # TODO FINE TUNE THIS PARAMETERS
            points_list = np.array(points[1])
            plane_points = np.array(aux_pc.points)[points_list]

            centers.append(np.mean(plane_points, axis=0))

            aux_pc = aux_pc.select_by_index(points[1], invert=True)
            # o3d.visualization.draw_geometries([aux_pc])  # TODO VISUALIZE DEBUG

        # TODO VISUALIZE DEBUG (VISUALIZE RECONSTRUCTED MESH AND FACE CENTERS)
        # centers_pc = PointCloud()  # TODO VISUALIZE DEBUG
        # centers_pc.points = Vector3dVector(np.array(centers))  # TODO VISUALIZE DEBUG
        # colors = np.array([255, 0, 0])[None, ...].repeat(len(centers_pc.points), axis=0)  # TODO VISUALIZE DEBUG
        # centers_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        # o3d.visualization.draw_geometries([rec_mesh, centers_pc], mesh_show_back_face=True)  # TODO VISUALIZE DEBUG

        # Get closest point from sampled point cloud for every center
        true_centers = []
        sampled_mesh = torch.FloatTensor(np.array(rec_pc.points))
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
            true_centers.append(sampled_mesh[torch.argmin(dist)].numpy())

        true_centers = np.array(true_centers).squeeze()

        # TODO VISUALIZE DEBUG (SHOWS RECONSTRUCTED MESH AND REAL CENTERS)
        true_pc = PointCloud()  # TODO VISUALIZE DEBUG
        true_pc.points = Vector3dVector(true_centers)
        colors = np.array([255, 0, 0])[None, ...].repeat(len(true_centers), axis=0)  # TODO VISUALIZE DEBUG
        true_pc.colors = Vector3dVector(colors)  # TODO VISUALIZE DEBUG
        rec_pc.estimate_normals()  # TODO VISUALIZE DEBUG
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([true_pc, rec_pc, coord], mesh_show_back_face=True)  # TODO VISUALIZE DEBUG


        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# List<String> free = new ArrayList<String>();
# for (Map.Entry < Integer, String > entry : this.Sala.entrySet()) {
#     if (!entry.isOccupato()){
#         free.add(entry.getNumer());
#     }
# Random rand = new Random();
# return free.get(rand.nextInt(free.size()));