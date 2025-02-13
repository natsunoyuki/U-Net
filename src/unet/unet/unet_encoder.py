import torch

from unet.unet.unet_blocks import DoubleConv2d, MaxPoolDoubleConv2d


class Encoder(torch.nn.Module):
    def __init__(
        self,
        channels=[1, 64, 128, 256, 512, 1024], kernel_size=3, stride=1, padding=1, bias=False,
        max_pool_kernel_size=2, max_pool_stride=None, max_pool_padding=0,
    ):
        super().__init__()
        self.blocks = []
        for i in range(len(channels)-1):
            if i == 0:
                bl = DoubleConv2d(channels[i], channels[i+1], kernel_size, stride, padding, bias)
            else:
                bl = MaxPoolDoubleConv2d(
                    channels[i], channels[i+1], kernel_size, stride, padding, bias,
                    max_pool_kernel_size, max_pool_stride, max_pool_padding,
                )
            self.blocks.append(bl)
        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs
