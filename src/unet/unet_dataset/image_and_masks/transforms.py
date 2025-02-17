import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
import torchvision.transforms.v2 as T2
import torch
import random


def get_transform(train: bool=False, image_size=[512, 512]):
    transforms = []

    if image_size is not None:
        transforms.append(Resize(image_size))

    transforms.append(PILToTensor())

    if train is True:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
        #transforms.append(T2.RandomAffine(90, [0, 0.1], [0.9, 1.1]))
        transforms.append(GaussianNoise(0.5, sigma = 0.1))

    return Compose(transforms)


class Compose:
    def __init__(self, transforms = []):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    

def pil_to_tensor(image):
    # F.pil_to_tensor does not scale the pixel values to [0, 1].
    image = F.pil_to_tensor(image)
    image = F.convert_image_dtype(image)
    return image


class PILToTensor(torch.nn.Module):
    def forward(self, image, mask = None):
        image = pil_to_tensor(image)
        if mask is not None:
            mask = pil_to_tensor(mask)
            mask = mask.round() # to {0, 1}.
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
    

class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, image: torch.Tensor, mask: torch.Tensor = None):
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)
        return image, mask
    

class RandomAffine(T2.RandomAffine):
    def __init__(self, p = 0, degrees = 0, translate = None, scale = None):
        super().__init__(degrees=degrees, translate=translate, scale=scale)
        self.p = p

    def forward(self, image: torch.Tensor, mask: torch.Tensor = None):
        if torch.rand(1) < self.p:
            self.transform(image, mask)
        return image, mask


class GaussianNoise(torch.nn.Module):
    def __init__(self, p = 0.5, mean=0, sigma=0.1, clip=True):
        super().__init__()
        self.p = p
        self.gaussian_noise = T2.GaussianNoise(mean, sigma, clip)
    
    def forward(self, image, mask = None):
        if torch.rand(1) < self.p:
            image = self.gaussian_noise(image)

        return image, mask