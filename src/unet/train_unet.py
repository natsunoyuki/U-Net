import torch

#import logging
#logger = logging.getLogger(__name__)

from unet.unet import UNet


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
    device = configs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    print("Using device: {}.".format(device))

    # U-Net.
    in_channels = configs.get("in_channels", 1)
    out_channels = configs.get("out_channels", 2)
    conv_channels = configs.get("conv_channels", [64, 128, 256, 512, 1024])
    
    unet = UNet(in_channels, out_channels, conv_channels)
    unet.to(device)
    unet.train()

    print("U-Net in_channels={}, out_channels={}, conv_channels={}.".format(
        in_channels, out_channels, conv_channels)
    )

    # Dataloaders.



    return
