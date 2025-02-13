"""
Script for making predictions with YOLOX models.
"""
import argparse
from pathlib import Path
import yaml
import os
import time
from loguru import logger

import cv2

import torch


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


class PredictConfigs:
    def __init__(self, yaml_args):
        self.demo = yaml_args.get("demo", "image")
        self.path = yaml_args.get("path", "./assets/dog.jpg")
        self.conf = yaml_args.get("conf", 0.25)
        self.nms = yaml_args.get("nms", 0.3)
        self.tsize = yaml_args.get("tsize", None)
        self.experiment_name = yaml_args.get("experiment_name", None)
        self.name = yaml_args.get("name", self.experiment_name)
        self.exp_file = yaml_args.get("exp_file", None)
        self.ckpt = yaml_args.get("ckpt", None)
        self.device = yaml_args.get("device", "cpu")
        self.camid = yaml_args.get("camid", 0)
        self.save_result = yaml_args.get("save_result", True)
        self.fp16 = yaml_args.get("fp16", False)
        self.legacy = yaml_args.get("legacy", False)
        self.fuse = yaml_args.get("fuse", False)
        self.trt = yaml_args.get("trt", False)


def get_image_list(path):
    image_names = []
    for maindir, _, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

