import torch

from unet.unet.unet_blocks import UpConvDoubleConv2d


class Decoder(torch.nn.Module):
    def __init__(
        self,	
        channels=[1024, 512, 256, 128, 64, 2], kernel_size=3, stride=1, padding=1, bias=False,
        up_conv_kernel_size=2, up_conv_stride=2, up_conv_padding=0, up_conv_bias=False, 
    ):
        super().__init__()
        self.blocks = []
        for i in range(len(channels[:-1])-1):
            bl = UpConvDoubleConv2d(
                 channels[i], channels[i+1], kernel_size, stride, padding, bias,
                 up_conv_kernel_size, up_conv_stride, up_conv_padding, up_conv_bias,
            )
            self.blocks.append(bl)
        self.blocks = torch.nn.ModuleList(self.blocks)

        self.output_segmentation_conv = torch.nn.Conv2d(channels[-2], channels[-1], kernel_size=1)

    def forward(self, encoder_outputs):
        x = encoder_outputs[-1]
        encoder_outputs = encoder_outputs[::-1][1:]
        for block, enc_x in zip(self.blocks, encoder_outputs):
            x = block(x, enc_x)

        x = self.output_segmentation_conv(x)
        return x
