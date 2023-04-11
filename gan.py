import torch
import torch.nn as nn
"""
Gan Modules needed for the implementation

Part 1 : U-Net


Part 2 : Discriminator


"""

# First the Conv Module

class conv_seq(nn.Module):
    
    def __init__(self, in_channels, out_channels,kernel_size):
        super(self,conv_seq).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# Now the U-Net Module
class unet(nn.Module):
    def __init__(self,in_channels,base_channels,out_channels):
        self.c_size = [base_channels,2*base_channels]
        self.input_conv = conv_seq(in_channels=in_channels,out_channels=base_channels,kernel_size=3)
        self.pool1 = nn.Max