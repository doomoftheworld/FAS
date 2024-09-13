"""""
Basic Network structures
"""""

from common_imports import *

# Convolutional layer
class Basic_Conv2d(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=nb_in_channels, out_channels=nb_out_channels, kernel_size=conv_k, stride=conv_stride, padding=conv_pad),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv2d(x)
        return x

# Maxpool Layer
class Basic_Maxpool2d(nn.Module):
    def __init__(self, pool_k, pool_stride, pool_pad):
        super().__init__()
        self.maxpool2d = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_k, stride=pool_stride, padding=pool_pad),
            # nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.maxpool2d(x)
        return x
    
class Basic_Conv2d_with_batch_norm(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad):
        super().__init__()
        self.conv2d_bn = nn.Sequential(
            nn.Conv2d(in_channels=nb_in_channels, out_channels=nb_out_channels, kernel_size=conv_k, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(nb_out_channels, eps=0.001),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv2d_bn(x)
        return x


# Convolutional block
class Conv_Block(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad, pool_k, pool_stride, pool_pad):
        super().__init__()
        self.conv = Basic_Conv2d(nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad)
        self.maxpool = Basic_Maxpool2d(pool_k, pool_stride, pool_pad)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x

# Convolutional block with batch normalization
class Conv_Block_with_batch_norm(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad, pool_k, pool_stride, pool_pad):
        super().__init__()
        self.conv = Basic_Conv2d_with_batch_norm(nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad)
        self.maxpool = Basic_Maxpool2d(pool_k, pool_stride, pool_pad)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x