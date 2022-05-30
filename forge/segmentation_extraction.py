import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

from utils.Segmentator import Segmentator
from utils.input import RealSense


def main():

    seg = models.segmentation.fcn_resnet101(pretrained=False)
    seg.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    seg.load_state_dict(torch.load('../Segmentation/checkpoints/sym5/epoch23'), strict=False)
    seg.eval()

    seg = Segmentator(seg, device='cuda')

    for i in range(2, 22):

        frame = np.array(Image.open(f'../Segmentation/data/real/data/rgb/eCub{i}_rgb.png'))
        depth = np.array(Image.open(f'../Segmentation/data/real/data/depth/eCub{i}_depth.png')).astype(np.uint16)

        # cv2.imshow('rgb', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # cv2.imshow('depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))
        # key = cv2.waitKey(0)

        x = seg.preprocess(frame).to(device='cuda')
        with torch.no_grad():
            y = seg.model(x)['out']
        score, categories = seg.postprocess(y)

        ### On the RGB
        _, small_categories = seg.postprocess(y, width=256, height=192)
        small_frame = cv2.resize(frame, dsize=(256, 192), interpolation=cv2.INTER_LINEAR)
        small_frame[small_categories != 1] = np.array([0, 0, 0])
        resized = cv2.resize(small_frame, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)

        frame[categories != 1] = np.array([0, 0, 0])

        # cv2.imshow('test_seg1', cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        # cv2.imshow('test_seg2', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #
        # cv2.waitKey(0)

        ### On the depth
        small_depth = cv2.resize(depth, dsize=(256, 192), interpolation=cv2.INTER_NEAREST)
        small_depth[small_categories != 1] = 0.0

        _, big_categories = seg.postprocess(y, width=800, height=600)
        big_depth = cv2.resize(depth, dsize=(800, 600), interpolation=cv2.INTER_NEAREST)
        big_depth[big_categories != 1] = 0.0

        depth[categories != 1] = 0.0

        small_pc = RealSense.pointcloud(small_depth, scale=0.4)
        big_pc = RealSense.pointcloud(big_depth, scale=1.25)
        pc = RealSense.pointcloud(depth)

        draw_pc(small_pc + np.array([1, 0, 0]), pc, big_pc + np.array([-1, 0, 0]))


from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries


def draw_pc(*pcs):
    o3d_pcs = []
    for pc in pcs:
        aux = PointCloud()
        aux.points = Vector3dVector(pc)
        aux.paint_uniform_color(np.random.rand(3))
        o3d_pcs.append(aux)

    draw_geometries(o3d_pcs)

if __name__ == '__main__':
    main()