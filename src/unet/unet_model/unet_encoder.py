import torch

from unet.unet_model.unet_blocks import DoubleConv2d, MaxPoolDoubleConv2d


class Encoder(torch.nn.Module):
    def __init__(
        self,
        channels=[3, 8, 16, 32], 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False,
        max_pool_kernel_size=2, 
        max_pool_stride=None, 
        max_pool_padding=0,
    ):
        super().__init__()

        # First input layer only has double convolutions without maxpooling.
        self.input_double_conv = DoubleConv2d(
            in_channels=channels[0], 
            out_channels=channels[1], 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias,
        )

        self.blocks = []
        for i in range(1, len(channels[1:])):
            # Maxpooling occurs only for the second layer onwards.
            b = MaxPoolDoubleConv2d(
                in_channels=channels[i], 
                out_channels=channels[i+1], 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                bias=bias,
                max_pool_kernel_size=max_pool_kernel_size, 
                max_pool_stride=max_pool_stride, 
                max_pool_padding=max_pool_padding,
            )
            self.blocks.append(b)
        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, x):
        x = self.input_double_conv(x)
        outputs = [x]
        
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # [block_0, block_1, block_2, ... block_n], where block_0 has a smaller 
        # number of channels but larger feature maps, and block_n has a larger
        # number of channels but smaller feature maps.
        return outputs
