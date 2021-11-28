###   it is just for research purpose, and commercial use is not allowed  ###

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
from torch.nn.modules.utils import _triple



class SpatioTemporalConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/(kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()   ##   nn.Tanh()   or   nn.ReLU(inplace=True)


        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):                              
        x = self.relu(self.bn(self.spatial_conv(x)))     
        x = self.temporal_conv(x)                       
        return x



# 
class STVEN_Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=4):
        super(STVEN_Generator, self).__init__()


        layers = []
        n_input_channel = 3+c_dim # image + label channels
        layers.append(nn.Conv3d(n_input_channel, conv_dim, kernel_size=(3,7,7), stride=(1,1,1), padding=(1,3,3), bias=False))
        layers.append(nn.InstanceNorm3d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim


        layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), bias=False))
        layers.append(nn.InstanceNorm3d(curr_dim*2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2

        layers.append(nn.Conv3d(curr_dim, curr_dim*4, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False))
        layers.append(nn.InstanceNorm3d(curr_dim*4, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 4

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(SpatioTemporalConv(curr_dim, curr_dim, [3, 3, 3], stride=(1,1,1), padding=[1,1,1]))
        
        # Up-sampling layers.
        layers2 = []
        layers2.append(nn.ConvTranspose3d(curr_dim, curr_dim//4, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False))
        layers2.append(nn.InstanceNorm3d(curr_dim//4, affine=True, track_running_stats=True))
        layers2.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim // 4
        
        layers3 = []
        layers3.append(nn.ConvTranspose3d(curr_dim, curr_dim//2, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False))
        layers3.append(nn.InstanceNorm3d(curr_dim//2, affine=True, track_running_stats=True))
        layers3.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim //2

        layers4 = []
        layers4.append(nn.Conv3d(curr_dim, 3, kernel_size=(1,7,7), stride=(1,1,1), padding=(0,3,3), bias=False))
        layers4.append(nn.Tanh())
        
        self.down3Dmain = nn.Sequential(*layers)
        
        self.layers2 = nn.Sequential(*layers2)
        self.layers3 = nn.Sequential(*layers3)
        self.layers4 = nn.Sequential(*layers4)
        

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1, 1)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3), x.size(4))

        x0 = torch.cat([x, c], dim=1)
        x0 = self.down3Dmain(x0)

        x1 = self.layers2(x0)
        x2 = self.layers3(x1)
        x3 = self.layers4(x2)

        out = x3 +x #Res Connection

        return out
    
