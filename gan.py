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

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding = 1)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding = 1)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Now the U-Net Module
class unet(nn.Module):

    def __init__(self,in_channels,base_channels,out_channels):

        self.input_conv = conv_seq(in_channels=in_channels,out_channels=base_channels,kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = conv_seq(in_channels=base_channels,out_channels=base_channels*2,kernel_size=3)
        self.conv2 = conv_seq(in_channels=base_channels*2,out_channels=base_channels*4,kernel_size=3)
        self.conv3 = conv_seq(in_channels=base_channels*4,out_channels=base_channels*8,kernel_size=3)
        self.conv4 = conv_seq(in_channels=base_channels*8,out_channels=base_channels*16,kernel_size=3)

        self.conv5 = conv_seq(in_channels=base_channels*16,out_channels=base_channels*8,kernel_size=3)
        self.conv6 = conv_seq(in_channels=base_channels*8,out_channels=base_channels*4,kernel_size=3)
        self.conv7 = conv_seq(in_channels=base_channels*4,out_channels=base_channels*2,kernel_size=3)
        self.conv8 = conv_seq(in_channels=base_channels*2,out_channels=base_channels,kernel_size=3)
        self.output_conv = nn.Conv2d(in_channels=base_channels, out_channels=self.out_channels, kernel_size=3, padding = 1)

    def forward(self,x):
        e1 = self.conv1(x)
        e2 = self.conv2(x)