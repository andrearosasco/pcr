###########################
# NOT USE THE MESH AT ALL #
###########################
import time
import numpy as np
import open3d as o3d
import torch
from main import HyperNetwork
from configs.server_config import ModelConfig
from frompartialtopose import iCubGazebo, FromPartialToPose, fp_sampling

device = "cuda"


if __name__ == '__main__':

    res = 0.01

    # Setting up environment
    icub = iCubGazebo()
    m = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
    m = m.to(device)
    m.eval()

    # TODO START TRY TO PUT CLASS
    generator = FromPartialToPose(m, res)
    # TODO END START TRY

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
        # key = cv2.waitKey(1) & 0xFF  # TODO VISUALIZE DEBUG
        # if key == ord("q"):  # TODO VISUALIZE DEBUG
        #     break  # TODO VISUALIZE DEBUG

        # Convert depth image to Point Cloud
        fx = fy = 343.12110728152936
        cx = 160.0
        cy = 120.0
        intrinsics = o3d.camera.PinholeCameraIntrinsic(320, 240, fx, fy, cx, cy)
        partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(filtered_depth), intrinsics)
        # o3d.visualization.draw_geometries([pc])  # TODO VISUALIZE DEBUG

        # Sample Point Cloud
        part = torch.FloatTensor(np.array(partial_pcd.points))  # Must be 1, 2024, 3
        start = time.time()  # TODO REMOVE DEBUG
        indices = fp_sampling(part.unsqueeze(0), 2024)  # TODO THIS IS SLOW
        print("fps took: {}".format(time.time() - start))  # TODO REMOVE DEBUG
        part = part[indices.long().squeeze()]

        # Normalize Point Cloud as training time
        part = np.array(part)
        mean = np.mean(np.array(part), axis=0)
        part = np.array(part) - mean
        var = np.sqrt(np.max(np.sum(part ** 2, axis=1)))
        part = part / (var * 2)

        part[..., -1] = -part[..., -1]  # TODO VERIFY (IS IT NORMAL THAT I NEED TO INVERT THIS?)

        complete = generator.reconstruct_point_cloud(part)
        p = generator.find_poses(complete, mult_res=4, n_points=10, iterations=100)
        coords = generator.orient_poses(p)
        o3d.visualization.draw_geometries(coords + [complete])
