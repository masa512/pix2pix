import torch
import torch.nn as nn
import numpy
import torchvision.models
"""
Gan Modules needed for the implementation

Part 1 : U-Net


Part 2 : Discriminator (Just use AlexNet)

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

class usamp_block(nn.Modue):
    
    def __init__(self, in_channels, out_channels):
        super(self,usamp_block).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.c_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, padding = 1)
        self.d_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding = 1)
    def forward(self,x,s):
        x = self.c_conv(self.upsample(x))
        y = self.d_conv(torch.concat([x,s], dim=1))
        return y



        

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

        self.output_conv = nn.Conv2d(in_channels=base_channels, out_channels=self.out_channels, kernel_size=1, padding = 0)

        self.us1 = usamp_block(base_channels*16, base_channels*8)
        self.us2 = usamp_block(base_channels*8, base_channels*4)
        self.us3 = usamp_block(base_channels*4, base_channels*2)
        self.us4 = usamp_block(base_channels*2, base_channels)


    def forward(self,x):
        x_in = self.input_conv(x)

        # Encoding
        e1 = self.conv1(self.pool(x_in))
        e2 = self.conv2(self.pool(e1))
        e3 = self.conv3(self.pool(e2))
        e4 = self.conv4(self.pool(e3))

        # Decoding
        y1 = self.us1(e4,e3)
        y2 = self.us2(y1,e2)
        y3 = self.us3(y2,e1)
        y4 = self.us3(y3,x_in)

        return self.output_conv(y4)
    

# Now time for a discriminator
