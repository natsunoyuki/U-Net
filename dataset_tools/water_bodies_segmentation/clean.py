from pathlib import Path
from PIL import Image
import numpy as np
import natsort

data_dir = Path("./")

images = sorted((data_dir / "Images/").iterdir())
masks = sorted((data_dir / "Masks/").iterdir())
for i in images:
    if i.name == ".DS_Store":
        images.remove(i)
for i in masks:
    if i.name == ".DS_Store":
        masks.remove(i)
images = sorted(images)
masks = sorted(masks)

print(len(images), len(masks))


for i, m in zip(images, masks):
    assert i.name == m.name
    img = Image.open(m)
    img = np.array(img)

    if (
        len(np.unique(img)) == 1 
        or img.shape[0] / img.shape[1] > 4/3 
        or img.shape[1] / img.shape[0] > 4/3
        or img.shape[0] < 512 
        or img.shape[1] < 512
    ):
        i.unlink()
        m.unlink()


images = sorted((data_dir / "Images/").iterdir())
masks = sorted((data_dir / "Masks/").iterdir())
for i in images:
    if i.name == ".DS_Store":
        images.remove(i)
for i in masks:
    if i.name == ".DS_Store":
        masks.remove(i)
images = natsort.natsorted(images)
masks = natsort.natsorted(masks)

print(len(images), len(masks))

