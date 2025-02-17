from pathlib import Path
from PIL import Image
import numpy as np
import natsort
import os
import random

random.seed(0)

def clean(images, masks, which="train"):
    for i, m in zip(images, masks):
        assert i.name.replace("sat_", "") == m.name.replace("mask_", "")

        image = Image.open(i).convert("RGB")
        mask = Image.open(m).convert("L")

        new_i = data_dir / "{}_images".format(which) / i.name.replace("sat_", "")
        new_m = data_dir / "{}_masks".format(which) / m.name.replace("mask_", "").replace(".jpg", ".png") 

        mask.save(new_m)
        image.save(new_i)


data_dir = Path("./data/forest_segmentation/")

image_dir = data_dir / "images/"
mask_dir = data_dir / "masks/"

images = sorted(image_dir.iterdir())
masks = sorted(mask_dir.iterdir())
for i in images:
    if i.name == ".DS_Store":
        images.remove(i)
for i in masks:
    if i.name == ".DS_Store":
        masks.remove(i)
images = natsort.natsorted(images)
masks = natsort.natsorted(masks)
print(len(images), len(masks))

indices = list(zip(images, masks))
random.shuffle(indices)
images, masks = zip(*indices)

N = int(0.8 * len(images))
train_images = images[:N]
test_images = images[N:]
train_masks = masks[:N]
test_masks = masks[N:]
print(len(train_images), len(train_masks), len(test_images), len(test_masks))




clean(train_images, train_masks, "train")
clean(test_images, test_masks, "test")


