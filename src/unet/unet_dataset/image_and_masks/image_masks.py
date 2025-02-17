from pathlib import Path
import torch
import natsort
from PIL import Image
import numpy as np

from unet.unet_dataset.file_utils import list_files
from unet.unet_dataset.image_and_masks.transforms import get_transform


class ImageMasksDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir: Path, 
        image_folder="images/", 
        mask_folder="masks/", 
        transforms=None,
        train=False,
        image_size=None,
    ):
        self.data_dir = data_dir
        self.image_folder = image_folder
        self.mask_folder = mask_folder

        self.image_files = list_files(data_dir / image_folder)
        self.mask_files = list_files(data_dir / mask_folder)

        self.train = train

        self.transforms = transforms
        if self.transforms is None:
            self.transforms = get_transform(train, image_size=image_size)


    def __getitem__(self, i):
        image_name = self.image_files[i]
        mask_name = self.mask_files[i]
        assert image_name.stem == mask_name.stem

        img = Image.open(image_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask


    def __len__(self):
        return len(self.mask_files)
    