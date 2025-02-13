# MS COCO format dataset.
from pathlib import Path
import torch
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

class MSCocoDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir, 
        ann_dir="annotations",
        ann_json_file="train.json",
        transforms=None,
    ):
        self.coco = COCO(data_dir / ann_dir / ann_json_file)








'''
class MSCocoDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir, 
        split="train", 
        annotations_folder="annotations",
        transforms=None,
    ):
        self.root = Path(data_dir)
        self.split = split
        self.transforms = transforms

        with open(self.root / annotations_folder / (split+".json"), "r") as f:
            self.annotations_dict = json.load(f)

        self.file_list = self.annotations_dict.get("images")
        self.ann_list = self.annotations_dict.get("annotations")

        self.img_id_ann_id = {}
        for i, a in enumerate(self.ann_list):
            if a["image_id"] not in self.img_id_ann_id:
                self.img_id_ann_id[a["image_id"]] = [i]
            else:
                self.img_id_ann_id[a["image_id"]].append(i)

    def __getitem__(self, i):
        img_path = self.root / self.split / self.file_list[i]["file_name"]
        img_id = self.root / self.split / self.file_list[i]["id"]

        img = np.array(Image.open(img_path).convert("RGB"))

        anns = self.img_id_ann_id[img_id]

        mask = np.zeros([*img.shape, len(anns)])

        for i in anns:
            smask = np.array(self.ann_list[i]["segmentation"])


        """
        # Load image from the hard disc.
        img = Image.open(os.path.join(self.root, 
              "images/" + self.files[i] + ".png")).convert("RGB")
        
        # Load annotation file from the hard disc.
        ann = xml_to_dict(os.path.join(self.root, 
              "annotations/" + self.files[i] + ".xml"))
        # The target is given as a dict.
        target = {}
        target["boxes"] = torch.as_tensor([[ann["x1"], 
                                            ann["y1"], 
                                            ann["x2"], 
                                            ann["y2"]]], 
                                   dtype = torch.float32)
        target["labels"]=torch.as_tensor([label_dict[ann["label"]]],
                         dtype = torch.int64)
        target["image_id"] = torch.as_tensor(i)
        # Apply any transforms to the data if required.
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    
        """
    def __len__(self):
        return len(self.file_list)
''';