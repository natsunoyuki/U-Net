from pathlib import Path
import torch

#import logging
#logger = logging.getLogger(__name__)

from unet.unet_model import UNet
from unet.unet_dataset.image_and_masks.image_masks import ImageMasksDataset


def train_unet(configs):
    """
    logging_handlers=[logging.StreamHandler()]
    log_file_path = configs.get("log_file_path", None)
    if log_file_path is not None:
        logging_handlers.append(logging.FileHandler(log_file_path))
        
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=logging_handlers,
    )
    """
    # Device.
    device = configs.get(
        "device", 
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device: {}.".format(device))

    # U-Net.
    in_channels = configs.get("in_channels", 3)
    out_channels = configs.get("out_channels", 1)
    conv_channels = configs.get("conv_channels", [64, 128, 256, 512, 1024])
    
    unet = UNet(in_channels, out_channels, conv_channels)
    unet.to(device)
    unet.train()

    print("U-Net in_channels={}, out_channels={}, conv_channels={}.".format(
        in_channels, out_channels, conv_channels)
    )

    # Dataloaders.
    data_dir = configs.get("data_dir", None)
    train_image_folder = configs.get("train_image_folder", "train_images")
    train_mask_folder = configs.get("train_mask_folder", "train_masks")
    test_image_folder = configs.get("test_image_folder", "test_images")
    test_mask_folder = configs.get("test_mask_folder", "test_masks")
    if data_dir is None:
        print("Data directory not specified. Terminating")
        return
    else:
        data_dir = Path(data_dir)
    # Train dataset. 
    train_ds = ImageMasksDataset(
        data_dir=data_dir, image_folder=train_image_folder, mask_folder=train_mask_folder, train=True
    )
    # Test dataset.
    test_ds = ImageMasksDataset(
        data_dir=data_dir, image_folder=test_image_folder, mask_folder=test_mask_folder, train=False
    )


    # Create the DataLoaders from the Datasets. 
    batch_size = configs.get("batch_size", 4)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size = batch_size, shuffle = True, collate_fn = collate_fn
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size = batch_size, shuffle = False, collate_fn = collate_fn
    )


    return


def collate_fn(batch):
    return tuple(zip(*batch))


def unbatch(batch, device):
    X, y = batch
    X = [x.to(device) for x in X]
    y = [{k: v.to(device) for k, v in t.items()} for t in y]
    return X, y


def train_batch(batch, model, optimizer, device):
    model.train()
    X, y = unbatch(batch, device = device)
    optimizer.zero_grad()
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses


@torch.no_grad()
def validate_batch(batch, model, optimizer, device):
    model.train()
    X, y = unbatch(batch, device = device)
    optimizer.zero_grad()
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())
    return loss, losses


def train(
    model, 
    optimizer, 
    n_epochs, 
    train_loader, 
    test_loader = None, 
    device = torch.device("cpu")
):
    model.to(device)

    for epoch in range(n_epochs):
        N = len(train_loader)
        for ix, batch in enumerate(train_loader):
            loss, losses = train_batch(batch, model, optimizer, device)

        if test_loader is not None:
            N = len(test_loader)
            for ix, batch in enumerate(test_loader):
                loss, losses = validate_batch(batch, model, optimizer, device)
    