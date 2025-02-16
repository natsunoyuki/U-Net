import torch
    

def pad_x1_to_x2_shape(x1, x2, mode="constant", value=0):
        x1_h, x1_w = x1.shape[2:]
        x2_h, x2_w = x2.shape[2:]
        d_h = x2_h - x1_h
        d_w = x2_w - x1_w
        p = [d_w // 2, d_w - d_w // 2, d_h // 2, d_h - d_h // 2]
        x = torch.nn.functional.pad(x1, p, mode=mode, value=value)
        return x


class Conv2d(torch.nn.Module):
    """UNet 2D convolution block."""
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConv2d(torch.nn.Module):
	"""UNet double 2d convolution block."""
	def __init__(
		self, in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False
	):
		super().__init__()
		self.conv2d1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
		self.conv2d2 = Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias)
		
	def forward(self, x):
		x = self.conv2d1(x)
		x = self.conv2d2(x)
		return x
	

class MaxPoolDoubleConv2d(torch.nn.Module):
    """Unet max pooling + double 2d convolution block."""
    def __init__(
        self, 
        in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False,
        max_pool_kernel_size=2, max_pool_stride=None, max_pool_padding=0,
    ):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(max_pool_kernel_size, max_pool_stride, max_pool_padding)
        self.conv2d = DoubleConv2d(in_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv2d(x)
        return x


class UpConvDoubleConv2d(torch.nn.Module):
    """Unet up convolution + double 2d convolution block."""
    def __init__(
        self, 
        in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False,
        up_conv_kernel_size=2, up_conv_stride=2, up_conv_padding=0, up_conv_bias=False, 
	):
        super().__init__()
        self.upconv = torch.nn.ConvTranspose2d(
            in_channels, in_channels // up_conv_kernel_size, 
            up_conv_kernel_size, up_conv_stride, up_conv_padding, bias=up_conv_bias,
        )
        self.conv2d = DoubleConv2d(in_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        if x1.shape != x2.shape:
            #print(x1.shape, x2.shape)
            x1 = pad_x1_to_x2_shape(x1, x2)
            #print(x1.shape, x2.shape)
        x1 = torch.cat([x1, x2], dim=1)
        x2 = self.conv2d(x1)
        return x2
