"""
Script for evaluating YOLOX models.
"""

import argparse
from pathlib import Path
import yaml
import os
import random
import warnings


class EvaluateConfigs:
    def __init__(self, yaml_args):
        self.experiment_name = yaml_args.get("experiment_name", None)
        self.name = yaml_args.get("name", self.experiment_name)

        self.dist_backend = yaml_args.get("dist_backend", "nccl")
        self.dist_url = yaml_args.get("dist_url", None)

        self.batch_size = yaml_args.get("batch_size", 16)
        self.devices = yaml_args.get("devices", None)
        self.num_machines = yaml_args.get("num_machines", 1)
        self.machine_rank = yaml_args.get("machine_rank", 0)
        
        self.exp_file = yaml_args.get("exp_file", None)

        self.ckpt = yaml_args.get("ckpt", None)

        self.conf = yaml_args.get("conf", 0.25)
        self.nms = yaml_args.get("nms", 0.3)
        self.tsize = yaml_args.get("tsize", None)
        self.seed = yaml_args.get("seed", None)

        self.fp16 = yaml_args.get("fp16", False)
        self.legacy = yaml_args.get("legacy", False)
        self.fuse = yaml_args.get("fuse", False)
        self.trt = yaml_args.get("trt", False)

        self.test = yaml_args.get("test", False)
        self.speed = yaml_args.get("speed", False)
        self.opts = yaml_args.get("opts", [])
