import os
# import yarp
import cv2
# import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from scipy.io import loadmat


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


class RealSense:
    def __init__(self, width=640, heigth=480):
        self.pipeline = rs.pipeline()
        configs = {}
        configs['device'] = 'Intel RealSense D435i'

        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)

        configs['depth'] = {'width': width, 'height': heigth, 'format': 'z16', 'fps': 30}
        configs['color'] = {'width': width, 'height': heigth, 'format': 'rgb8', 'fps': 30}

        HIGH_ACCURACY = 3
        HIGH_DENSITY = 4
        MEDIUM_DENSITY = 5
        self.profile.get_device().sensors[0].set_option(rs.option.visual_preset, HIGH_DENSITY)

        configs['options'] = {}
        for device in self.profile.get_device().sensors:
            configs['options'][device.name] = {}
            for option in device.get_supported_options():
                configs['options'][device.name][str(option)[7:]] = str(device.get_option(option))

        self.configs = configs
        self.align = rs.align(rs.stream.depth)

    def intrinsics(self):
        return self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        # color_frame = aligned_frames.get_color_frame()
        depth_frame = frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    @classmethod
    def pointcloud(cls, depth_image, rgb_image=None):
        if rgb_image is None:
            return cls._pointcloud(depth_image)

        depth_image = o3d.geometry.Image(depth_image)
        rgb_image = o3d.geometry.Image(rgb_image)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(rgb_image, depth_image,
                                                                    convert_rgb_to_intensity=False,
                                                                    depth_scale=1000)

        intrinsics = {'fx': 384.025146484375, 'fy': 384.025146484375, 'ppx': 319.09661865234375,
                      'ppy': 237.75723266601562,
                      'width': 640, 'height': 480}

        camera = o3d.camera.PinholeCameraIntrinsic(intrinsics['width'], intrinsics['height'], intrinsics['fx'],
                                                   intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        return pcd

    @classmethod
    def _pointcloud(cls, depth_image):
        depth_image = o3d.geometry.Image(depth_image)

        intrinsics = {'fx': 384.025146484375, 'fy': 384.025146484375, 'ppx': 319.09661865234375, 'ppy': 237.75723266601562,
                      'width': 640, 'height': 480}
        camera = o3d.camera.PinholeCameraIntrinsic(intrinsics['width'], intrinsics['height'], intrinsics['fx'],
                                                   intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])

        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, camera)
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        return np.array(pcd.points)

    def stop(self):
        self.pipeline.stop()


class YCBVideoReader:
    def __init__(self, data_path="/home/IIT.LOCAL/arosasco/projects/DenseFusion/datasets/ycb/YCB_Video_Dataset"):
        self.data_path = os.path.join(data_path, "data")
        self.video_list = os.listdir(self.data_path)
        self.video_id = 0
        self.frame_id = 1

    def get_frame(self):

        # Check if dataset is over
        if self.video_id > len(self.video_list):
            return None

        # Create right path
        video_path = os.path.join(self.data_path, self.video_list[self.video_id])
        str_id = str(self.frame_id)
        str_id = '0' * (6 - len(str_id)) + str_id
        frame_path = os.path.join(video_path, str_id)

        # Open bounding boxes
        boxes_path = frame_path + '-box.txt'
        boxes = {}
        with open(boxes_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            boxes[line[0]] = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]

        # Open rgb image
        rgb_path = frame_path + '-color.png'
        rgb = cv2.imread(rgb_path)

        # Open depth image
        depth_path = frame_path + '-depth.png'
        depth = cv2.imread(depth_path)

        # Open label image
        label_path = frame_path + '-label.png'
        label = cv2.imread(label_path)

        # Open
        mat_path = frame_path + '-meta.mat'
        meta = loadmat(mat_path)

        # Next frame (or next video)
        self.frame_id += 1
        str_id = str(self.frame_id)
        str_id = '0' * (6 - len(str_id)) + str_id
        frame_path = os.path.join(video_path, str_id)
        if not os.path.exists(frame_path + '-box.txt'):
            self.frame_id = 1
            self.video_id += 1

        return frame_path, boxes, rgb, depth, label, meta


if __name__ == '__main__':
    camera = RealSense()

    while True:
        rgb, depth = camera.read()

        cv2.imshow('rgb', rgb)
        cv2.imshow('depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))

        cv2.waitKey(1)
