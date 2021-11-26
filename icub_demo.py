import yarp
import cv2
import numpy as np
import open3d as o3d
import torch


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
    icub = iCubGazebo()
    while True:
        rgb, depth = icub.read()
        cv2.imshow('RGB', rgb)
        cv2.imshow('Depth', depth)

        # Get only red part
        rgb_mask = rgb[..., 2] == 102  # Red is the last dimension
        rgb_mask = rgb_mask.astype(float) * 255
        cv2.imshow('Mask', rgb_mask)

        # Get only depth of the box
        filtered_depth = np.where(rgb_mask, depth, 0.)
        filtered_depth_img = filtered_depth.astype(float) * 255
        cv2.imshow('Filtered Depth', filtered_depth_img)

        # Convert depth image to Point Cloud
        fx = fy = 343.12110728152936
        cx = 160.0
        cy = 120.0
        intrinsics = o3d.camera.PinholeCameraIntrinsic(320, 240, fx, fy, cx, cy)
        pc = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(filtered_depth), intrinsics)

        partial = torch.FloatTensor(np.array(pc.points))

        # o3d.visualization.draw_geometries([pc])  TODO VISUALIZE POINT CLOUD

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
