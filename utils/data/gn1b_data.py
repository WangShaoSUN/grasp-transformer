import logging
import os
import copy
import glob

import torch
import torch.utils.data
from graspnetAPI import GraspNet
from .grasp_data import GraspDatasetBase
from .cornell_data import CornellDataset

from utils.dataset_processing import grasp, image

graspnet_root = "/home/zzl/Pictures/graspnet"


class GraspNet1BDataset(GraspDatasetBase):

    def __init__(self, file_path, camera='realsense', split='train', scale=2.0, ds_rotate=True,
                                output_size=224,
                                random_rotate=True, random_zoom=True,
                                include_depth=True,
                                include_rgb=True,
                                ):
        super(GraspNet1BDataset, self).__init__(output_size=output_size, include_depth=include_depth, include_rgb=include_rgb, random_rotate=random_rotate,
                 random_zoom=random_zoom, input_only=False)
        logging.info('Graspnet root = {}'.format(graspnet_root))
        logging.info('Using data from camera {}'.format(camera))
        self.graspnet_root = graspnet_root
        self.camera = camera
        self.split = split
        self.scale = scale  # 原图是hxw=720x1280，scale将原图缩小scale倍

        self._graspnet_instance = GraspNet(graspnet_root, camera, split)

        self.g_rgb_files = self._graspnet_instance.rgbPath  # 存放rgb的路径
        self.g_depth_files = self._graspnet_instance.depthPath  # 存放深度图的路径
        self.g_rect_files = []  # 存放抓取标签的路径

        for original_rect_grasp_file in self._graspnet_instance.rectLabelPath:
            self.g_rect_files.append(
                original_rect_grasp_file
                    .replace('rect', 'rect_cornell')
                    .replace('.npy', '.txt')
            )

        logging.info('Graspnet 1Billion dataset created!!')

    def _get_crop_attrs(self, idx, return_gtbbs=False):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.g_rect_files[idx], scale=self.scale)
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, int(1280 // self.scale) - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, int(720 // self.scale) - self.output_size))
        if not return_gtbbs:
            return center, left, top
        else:
            return center, left, top, gtbbs

    def get_gtbb(self, idx, rot=0, zoom=1):
        # gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.g_rect_files[idx], scale=self.scale)
        center, left, top, gtbbs = self._get_crop_attrs(idx, return_gtbbs=True)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        # graspnet 1b中的深度图单位转换成m
        depth_img = image.DepthImage.from_tiff(self.g_depth_files[idx], depth_scale=1000.0)
        rh, rw = int(720 // self.scale), int(1280 // self.scale)
        # 读入的是wxh=1280x720 resize成目标尺寸
        depth_img.resize((rh, rw))
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(rh, top + self.output_size), min(rw, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.g_rgb_files[idx])
        rh, rw = int(720 // self.scale), int(1280 // self.scale)
        # 读入的是wxh=1280x720 resize成目标尺寸
        rgb_img.resize((rh, rw))
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(rh, top + self.output_size), min(rw, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def __len__(self):
        return len(self.g_rect_files)