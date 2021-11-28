# We train rPPGNet for 15 epochs
# Adam optimizer is used while learning rate is set to 1e−4.
# For all facial videos, we use the
# Viola-Jones face detector [33] to detect and crop the coarse
# face area (see Figure 8 (a)) and remove background. We
# generate binary skin masks by open source Bob1 with
# threshold=0.3 as the ground truth. All face and skin images
# are normalized to 128x128 and 64x64 respectively
import math
import torch.nn as nn
from torch.nn.modules.utils import _triple
import pdb
import torch


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
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/(
            kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # self-definition
        #intermed_channels = int((in_channels+intermed_channels)/2)

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()  # nn.Tanh()   or   nn.ReLU(inplace=True)

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


class MixA_Module(nn.Module):
    """ Spatial-Skin attention module"""

    def __init__(self):
        super(MixA_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.AVGpool = nn.AdaptiveAvgPool1d(1)
        self.MAXpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, skin):
        """
            inputs :
                x : input feature maps( B X C X T x W X H)
                skin : skin confidence maps( B X T x W X H)
            returns :
                out : attention value
                spatial attention: W x H
        """
        m_batchsize, C, T, W, H = x.size()
        B_C_TWH = x.view(m_batchsize, C, -1)
        B_TWH_C = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        B_TWH_C_AVG = torch.sigmoid(self.AVGpool(
            B_TWH_C)).view(m_batchsize, T, W, H)
        B_TWH_C_MAX = torch.sigmoid(self.MAXpool(
            B_TWH_C)).view(m_batchsize, T, W, H)
        B_TWH_C_Fusion = B_TWH_C_AVG + B_TWH_C_MAX + skin
        Attention_weight = self.softmax(
            B_TWH_C_Fusion.view(m_batchsize, T, -1))
        Attention_weight = Attention_weight.view(m_batchsize, T, W, H)
        # mask1 mul
        output = x.clone()
        for i in range(C):
            output[:, i, :, :, :] = output[:, i, :, :, :].clone() * \
                Attention_weight

        return output, Attention_weight


# for open-source
# skin segmentation + PhysNet + MixA3232 + MixA1616part4
class rPPGNet(nn.Module):
    def __init__(self, frames=64):
        super(rPPGNet, self).__init__()

        self.ConvSpa1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvSpa3 = nn.Sequential(
            SpatioTemporalConv(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa4 = nn.Sequential(
            SpatioTemporalConv(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvSpa5 = nn.Sequential(
            SpatioTemporalConv(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa6 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa7 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa8 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa9 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvSpa10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.ConvSpa11 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.ConvPart1 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.ConvPart2 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.ConvPart3 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.ConvPart4 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.AvgpoolSpa = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.AvgpoolSkin_down = nn.AvgPool2d((2, 2), stride=2)
        self.AvgpoolSpaTem = nn.AvgPool3d((2, 2, 2), stride=2)

        self.ConvSpa = nn.Conv3d(3, 16, [1, 3, 3], stride=1, padding=[0, 1, 1])

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.poolspa = nn.AdaptiveAvgPool3d(
            (frames, 1, 1))    # attention to this value

        # skin_branch
        self.skin_main = nn.Sequential(
            nn.Conv3d(32, 16, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.skin_residual = nn.Sequential(
            nn.Conv3d(32, 8, [1, 1, 1], stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.skin_output = nn.Sequential(
            nn.Conv3d(8, 1, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.Sigmoid(),  # binary
        )

        self.MixA_Module = MixA_Module()

    def forward(self, x):	    	# x [3, 64, 128,128]
        x_visual = x

        x = self.ConvSpa1(x)		     # x [3, 64, 128,128]
        x = self.AvgpoolSpa(x)       # x [16, 64, 64,64]

        x = self.ConvSpa3(x)		    # x [32, 64, 64,64]
        x_visual6464 = self.ConvSpa4(x)	    	# x [32, 64, 64,64]
        x = self.AvgpoolSpa(x_visual6464)      # x [32, 64, 32,32]

        # branch 1: skin segmentation
        x_skin_main = self.skin_main(x_visual6464)    # x [8, 64, 64,64]
        x_skin_residual = self.skin_residual(x_visual6464)   # x [8, 64, 64,64]
        x_skin = self.skin_output(
            x_skin_main+x_skin_residual)    # x [1, 64, 64,64]
        x_skin = x_skin[:, 0, :, :, :]    # x [74, 64,64]

        # branch 2: rPPG
        x = self.ConvSpa5(x)		    # x [64, 64, 32,32]
        x_visual3232 = self.ConvSpa6(x)	    	# x [64, 64, 32,32]
        x = self.AvgpoolSpa(x_visual3232)      # x [64, 64, 16,16]

        x = self.ConvSpa7(x)		    # x [64, 64, 16,16]
        x = self.ConvSpa8(x)	    	# x [64, 64, 16,16]
        x_visual1616 = self.ConvSpa9(x)	    	# x [64, 64, 16,16]

        # SkinA1_loss
        x_skin3232 = self.AvgpoolSkin_down(x_skin)          # x [64, 32,32]
        x_visual3232_SA1, Attention3232 = self.MixA_Module(
            x_visual3232, x_skin3232)
        x_visual3232_SA1 = self.poolspa(x_visual3232_SA1)     # x [64, 64, 1,1]
        ecg_SA1 = self.ConvSpa10(x_visual3232_SA1).squeeze(
            1).squeeze(-1).squeeze(-1)

        # SkinA2_loss
        x_skin1616 = self.AvgpoolSkin_down(x_skin3232)       # x [64, 16,16]
        x_visual1616_SA2, Attention1616 = self.MixA_Module(
            x_visual1616, x_skin1616)
        # Global
        global_F = self.poolspa(x_visual1616_SA2)     # x [64, 64, 1,1]
        ecg_global = self.ConvSpa11(global_F).squeeze(
            1).squeeze(-1).squeeze(-1)

        # Local
        Part1 = x_visual1616_SA2[:, :, :, :8, :8]
        Part1 = self.poolspa(Part1)     # x [64, 64, 1,1]
        ecg_part1 = self.ConvSpa11(Part1).squeeze(1).squeeze(-1).squeeze(-1)

        Part2 = x_visual1616_SA2[:, :, :, 8:16, :8]
        Part2 = self.poolspa(Part2)     # x [64, 64, 1,1]
        ecg_part2 = self.ConvPart2(Part2).squeeze(1).squeeze(-1).squeeze(-1)

        Part3 = x_visual1616_SA2[:, :, :, :8, 8:16]
        Part3 = self.poolspa(Part3)     # x [64, 64, 1,1]
        ecg_part3 = self.ConvPart3(Part3).squeeze(1).squeeze(-1).squeeze(-1)

        Part4 = x_visual1616_SA2[:, :, :, 8:16, 8:16]
        Part4 = self.poolspa(Part4)     # x [64, 64, 1,1]
        ecg_part4 = self.ConvPart4(Part4).squeeze(1).squeeze(-1).squeeze(-1)

        return x_skin, ecg_SA1, ecg_global, ecg_part1, ecg_part2, ecg_part3, ecg_part4, x_visual6464, x_visual3232
