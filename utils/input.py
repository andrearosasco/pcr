import os
import cv2
import pyrealsense2 as rs

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.io import loadmat


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
    def pointcloud(cls, depth_image, rgb_image=None, scale=1.0):
        if rgb_image is None:
            return cls._pointcloud(depth_image, scale)

        depth_image = o3d.geometry.Image(depth_image)
        rgb_image = o3d.geometry.Image(rgb_image)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(rgb_image, depth_image,
                                                                    convert_rgb_to_intensity=False,
                                                                    depth_scale=1000)

        # intrinsics = {'fx': 384.025146484375, 'fy': 384.025146484375, 'ppx': 319.09661865234375,
        #               'ppy': 237.75723266601562,
        #               'width': 640, 'height': 480}

        intrinsics = {'fx': 612.7910766601562, 'fy': 611.8779296875, 'ppx': 321.7364196777344,
                      'ppy': 245.0658416748047,
                      'width': 640, 'height': 480}

        intrinsics.update((key, intrinsics[key] * 1.0) for key in intrinsics)

        camera = o3d.camera.PinholeCameraIntrinsic(int(intrinsics['width']), int(intrinsics['height']), intrinsics['fx'],
                                                   intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        return pcd

    @classmethod
    def _pointcloud(cls, depth_image, scale=1.0):
        depth_image = o3d.geometry.Image(depth_image)

        # intrinsics = {'fx': 384.025146484375, 'fy': 384.025146484375, 'ppx': 319.09661865234375, 'ppy': 237.75723266601562,
        #               'width': 640, 'height': 480}

        intrinsics = {'fx': 612.7910766601562, 'fy': 611.8779296875, 'ppx': 321.7364196777344,
                      'ppy': 245.0658416748047,
                      'width': 640, 'height': 480}

        intrinsics.update((key, intrinsics[key] * scale) for key in intrinsics)

        camera = o3d.camera.PinholeCameraIntrinsic(int(intrinsics['width']), int(intrinsics['height']), intrinsics['fx'],
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
        self.root_path = data_path
        self.data_path = os.path.join(data_path, "data")
        self.video_list = os.listdir(self.data_path)
        self.video_id = 0
        self.frame_id = 1
        self.intrinsics_1 = {
            'fx': 1066.778, 'fy': 1067.487,
            'cx': 312.9869, 'cy': 241.3109,
        }

        self.intrinsics_2 = {
            'fx': 1077.836, 'fy': 1078.189,
            'cx': 323.7872, 'cy': 279.6921,
        }
        self.jump_n_frames = 1000

    def get_xyz_by_name(self, name):
        path = os.path.join(self.root_path, "models")
        obj_path = os.path.join(path, name)
        obj_path = os.path.join(obj_path, 'points.xyz')
        points = np.genfromtxt(obj_path, delimiter=' ')
        return points

    def get_mesh_path_by_name(self, name):
        path = os.path.join(self.root_path, "models")
        obj_path = os.path.join(path, name)
        obj_path = os.path.join(obj_path, 'textured.obj')
        return obj_path

    def get_frame(self):

        # Check if dataset is over
        if self.video_id >= len(self.video_list):
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
        depth = Image.open(depth_path)
        depth = np.array(depth)

        # Open label image
        label_path = frame_path + '-label.png'
        label = cv2.imread(label_path)

        # Open meta
        mat_path = frame_path + '-meta.mat'
        meta = loadmat(mat_path)

        # Select appropriate intrinsics
        # is_syn = len(self.frames[idx].split('/')) == 2  TODO ADD ALSO THIS
        if self.video_id <= 59:
            intrinsics = self.intrinsics_1
        else:
            intrinsics = self.intrinsics_2

        # Next frame (or next video)
        self.frame_id += self.jump_n_frames
        str_id = str(self.frame_id)
        str_id = '0' * (6 - len(str_id)) + str_id
        frame_path = os.path.join(video_path, str_id)
        if not os.path.exists(frame_path + '-box.txt'):
            self.frame_id = 1
            self.video_id += 1

        # Remove last dimension from label (redundant)
        label = label[..., 0]

        return frame_path, boxes, rgb, depth, label, meta, intrinsics


if __name__ == '__main__':
    camera = RealSense()

    while True:
        rgb, depth = camera.read()

        cv2.imshow('rgb', rgb)
        cv2.imshow('depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))

        cv2.waitKey(1)