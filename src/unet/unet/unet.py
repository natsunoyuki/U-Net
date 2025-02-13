import torch

from unet.unet.unet_encoder import Encoder
from unet.unet.unet_decoder import Decoder


class UNet(torch.nn.Module):
    """UNet model described in https://arxiv.org/pdf/1505.04597."""
    def __init__(
        self, in_channels=1, out_channels=2, conv_channels=[64, 128, 256, 512, 1024]
    ):
        super().__init__()
        self.encoder = Encoder(
            channels=[in_channels] + conv_channels, kernel_size=3, stride=1, padding=1, bias=False,
            max_pool_kernel_size=2, max_pool_stride=None, max_pool_padding=0
        )
        self.decoder = Decoder(
            channels=conv_channels[::-1]+[out_channels], kernel_size=3, stride=1, padding=1, bias=False,
            up_conv_kernel_size=2, up_conv_stride=2, up_conv_padding=0, up_conv_bias=False, 
        )

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        x = self.decoder(encoder_outputs) 
        return x
        
