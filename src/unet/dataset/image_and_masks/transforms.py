import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
import torch


def get_transform(train: bool = False):
    """
    Transforms a PIL Image into a torch tensor, and performs
    random horizontal flipping of the image if training a model.
    Inputs
        train: bool
            Flag indicating whether model training will occur.
    Returns
        compose: Compose
            Composition of image transforms.
    """
    transforms = []
    # ToTensor is applied to all images.
    transforms.append(PILToTensor())
    # The following transforms are applied only to the train set.
    if train is True:
        transforms.append(RandomHorizontalFlip(0.5))
        # Other transforms can be added here later on.
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
            mask = mask.round().to(int)
        return image, mask


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: torch.Tensor, mask: torch.Tensor = None):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask
    