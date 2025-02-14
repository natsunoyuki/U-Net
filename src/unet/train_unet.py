from pathlib import Path
import torch

#import logging
#logger = logging.getLogger(__name__)

from unet.unet_model import UNet
from unet.unet_dataset.image_and_masks.image_masks import ImageMasksDataset
from unet.train_unet_utils import collate_fn, train


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
    if torch.cuda.is_available() is False and device == torch.device("cuda"):
        device = torch.device("cpu")
    print("Using device: {}.".format(device))

    # U-Net.
    in_channels = configs.get("in_channels", 3)
    out_channels = configs.get("out_channels", 1)
    conv_channels = configs.get("conv_channels", [64, 128, 256, 512, 1024])
    
    model = UNet(in_channels, out_channels, conv_channels)
    model.to(device)
    model.train()

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
    n_epochs = configs.get("epochs", 100)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size = batch_size, shuffle = True, collate_fn = collate_fn
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size = batch_size, shuffle = False, collate_fn = collate_fn
    )


    # Set up optimizer.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005
    )

    model, train_losses, test_losses = train(
        model, optimizer, n_epochs, train_dl, test_dl, device
    )

    return
