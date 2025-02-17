from pathlib import Path
from PIL import Image
import numpy as np
import natsort
import tqdm


def repair_masks(images, masks):
    for i, m in tqdm.tqdm(zip(images, masks)):
        assert i.name == m.name

        image = np.array(Image.open(i).convert("RGB"))
        mask = np.array(Image.open(m).convert("L"))
        mask[image[:, :, 0]==0] = 0
        mask[image[:, :, 1]==0] = 0
        mask[image[:, :, 2]==0] = 0
        mask[mask < 127] == 0
        mask[mask >= 127] == 0

        mask = Image.fromarray(mask)
        mask.save(m.rename(m.with_suffix('.png')))


data_dir = Path("./")

def list_files(path):
    files = list(path.iterdir())
    for i in files:
        if i.name.startswith("."):
            files.remove(i)
    return files


train_images = list_files(data_dir / "train_images")
train_masks = list_files(data_dir / "train_masks")
test_images = list_files(data_dir / "test_images")
test_masks = list_files(data_dir / "test_masks")


repair_masks(train_images, train_masks)
repair_masks(test_images, test_masks)
