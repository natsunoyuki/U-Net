"""
Script for training YOLOX models.
"""

from pathlib import Path
import argparse
import yaml

import random
import warnings



class TrainConfigs:
    def __init__(self, yaml_args):
        self.experiment_name = yaml_args.get("experiment_name", None)
        self.name = yaml_args.get("name", self.experiment_name)
        self.exp_file = yaml_args.get("exp_file", None)
        self.batch_size = yaml_args.get("batch_size", 16)
        self.devices = yaml_args.get("devices", None)
        self.resume = yaml_args.get("resume", False)
        self.ckpt = yaml_args.get("ckpt", None)
        self.start_epoch = yaml_args.get("start_epoch", None)
        self.fp16 = yaml_args.get("fp16", True)
        self.num_machines = yaml_args.get("num_machines", 1)
        self.machine_rank = yaml_args.get("machine_rank", 0)
        self.cache = yaml_args.get("cache", "ram")
        self.occupy = yaml_args.get("occupy", False)
        self.logger = yaml_args.get("logger", "tensorboard")
        self.opts = yaml_args.get("opts", [])
        self.dist_backend = yaml_args.get("dist_backend", "nccl")
        self.dist_url = yaml_args.get("dist_url", None)

