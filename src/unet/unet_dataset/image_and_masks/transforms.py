import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
import torch


def get_transform(train: bool=False, image_size=[512, 512]):
    transforms = []
    if image_size is not None:
        transforms.append(Resize(image_size))

    transforms.append(PILToTensor())
    if train is True:
        transforms.append(RandomHorizontalFlip(0.5))

    return Compose(transforms)


class Compose:
    def __init__(self, transforms = []):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    

def pil_to_tensor(image):
    image = F.pil_to_tensor(image)
    image = F.convert_image_dtype(image)
    return image


class PILToTensor(torch.nn.Module):
    def forward(self, image, mask = None):
        image = pil_to_tensor(image)
        if mask is not None:
            mask = pil_to_tensor(mask)
            mask = mask.round()
        return image, mask


class Resize(torch.nn.Module):
    def __init__(self, new_size = [512, 512]):
        super().__init__()
        self.resize = T.Resize(new_size)

    def forward(self, image, mask = None):
        image = self.resize(image)
        if mask is not None:
            mask = self.resize(mask)
        return image, mask


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: torch.Tensor, mask: torch.Tensor = None):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask
    