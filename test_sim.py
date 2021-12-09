import msvcrt
import open3d as o3d
import numpy as np
import torch
from frompartialtopose import iCubGazebo, GenPose, fp_sampling
import cv2

if __name__ == "__main__":
    test = GenPose()
    icub = iCubGazebo()

    test.reset_coords()

    while True:

        # Get image
        rgb, depth = icub.read()
        cv2.imshow('RGB', rgb)  # TODO VISUALIZE DEBUG

        # Get only red part
        rgb_mask = rgb[..., 2] == 102  # Red is the last dimension
        rgb_mask = rgb_mask.astype(float) * 255

        # Get only depth of the box
        filtered_depth = np.where(rgb_mask, depth, 0.)
        filtered_depth_img = filtered_depth.astype(float) * 255

        # Convert depth image to Point Cloud
        fx = fy = 343.12110728152936
        cx = 160.0
        cy = 120.0
        intrinsics = o3d.camera.PinholeCameraIntrinsic(320, 240, fx, fy, cx, cy)
        partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(filtered_depth), intrinsics)
        # o3d.visualization.draw_geometries([pc])  # TODO VISUALIZE DEBUG

        # Sample Point Cloud
        part = torch.FloatTensor(np.array(partial_pcd.points))  # Must be 1, 2024, 3
        # indices = fp_sampling(part.unsqueeze(0), 2024)  # TODO THIS IS SLOW
        # part = part[indices.long().squeeze()]  # TODO THIS IS SLOW
        part = part[torch.randperm(part.size()[0])]  # TODO THIS IS FAST BUT LESS ACCURATE (?)
        part = part[:2024]  # TODO THIS IS FAST BUT LESS ACCURATE (?)

        # Normalize Point Cloud as training time
        part = np.array(part)
        mean = np.mean(np.array(part), axis=0)
        part = np.array(part) - mean
        var = np.sqrt(np.max(np.sum(part ** 2, axis=1)))
        part = part / (var * 2)

        part[..., -1] = -part[..., -1]  # TODO VERIFY (IS IT NORMAL THAT I NEED TO INVERT THIS?)

        test.run(part)

        if msvcrt.kbhit():
            print(msvcrt.getch())
            test.reset_coords()

