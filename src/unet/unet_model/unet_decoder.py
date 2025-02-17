import torch

from unet.unet_model.unet_blocks import UpConvDoubleConv2d


class Decoder(torch.nn.Module):
    def __init__(
        self,	
        channels=[32, 16, 8, 1], 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False,
        up_conv_by_resampling=True,
        up_conv_kernel_size=2,
        up_conv_stride=2, 
        up_conv_padding=0, 
        up_conv_bias=False, 
    ):
        super().__init__()

        # The first part of the decoder up-convolves feature maps from the 
        # previous layers of the encoder.
        self.blocks = []
        for i in range(len(channels[:-1])-1):
            b = UpConvDoubleConv2d(
                in_channels=channels[i], 
                out_channels=channels[i+1], 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                bias=bias,
                up_conv_by_resampling=up_conv_by_resampling,
                up_conv_kernel_size=up_conv_kernel_size, 
                up_conv_stride=up_conv_stride, 
                up_conv_padding=up_conv_padding, 
                up_conv_bias=up_conv_bias,
            )
            self.blocks.append(b)
        self.blocks = torch.nn.ModuleList(self.blocks)

        # Output convolution has no up-convolution operations.
        self.output_segmentation_conv = torch.nn.Conv2d(
            channels[-2], 
            channels[-1], 
            kernel_size=1,
        )

    def forward(self, encoder_outputs):
        # encoder_outputs = [block_0, block_1, block_2, ... block_n], where 
        # block_0 has a smaller number of channels but larger feature maps, and 
        # block_n has a larger number of channels but smaller feature maps.
        x = encoder_outputs[-1]
        encoder_outputs = encoder_outputs[::-1][1:]

        for block, enc_x in zip(self.blocks, encoder_outputs):
            x = block(x, enc_x)

        x = self.output_segmentation_conv(x)
        return x
