import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from mamba_ssm import Mamba
from torch.nn import functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x*attention

class LateralConnection(nn.Module):
    def __init__(self, fast_channels=32, slow_channels=64):
        super(LateralConnection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, [3, 1, 1], stride=[2, 1, 1], padding=[1,0,0]),   
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        
    def forward(self, slow_path, fast_path):
        fast_path = self.conv(fast_path)
        return fast_path + slow_path

class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal
    
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super(MambaLayer, self).__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        drop_path = 0
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_patch_token(self, x):
        B, C, nf, H, W = x.shape
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))
        out = x_out.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out 

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        out = self.forward_patch_token(x)
        return out

def conv_block(in_channels, out_channels, kernel_size, stride, padding, bn=True, activation='relu'):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm3d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)


class PhysMamba(nn.Module):
    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128):
        super(PhysMamba, self).__init__()

        self.ConvBlock1 = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])  
        self.ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)
        self.ConvBlock6 = conv_block(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0], activation='elu')

        # Temporal Difference Mamba Blocks
        # Slow Stream
        self.Block1 = self._build_block(64, theta)
        self.Block2 = self._build_block(64, theta)
        self.Block3 = self._build_block(64, theta)
        # Fast Stream
        self.Block4 = self._build_block(32, theta)
        self.Block5 = self._build_block(32, theta)
        self.Block6 = self._build_block(32, theta)

        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(96, 48, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(48),
            nn.ELU(),
        )

        self.ConvBlockLast = nn.Conv3d(48, 1, [1, 1, 1], stride=1, padding=0)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.fuse_1 = LateralConnection(fast_channels=32, slow_channels=64)
        self.fuse_2 = LateralConnection(fast_channels=32, slow_channels=64)

        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def _build_block(self, channels, theta):
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            MambaLayer(dim=channels),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )
    
    def forward(self, x): 
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x) 
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)  
        x = self.MaxpoolSpa(x) 
    
        # Process streams
        s_x = self.ConvBlock4(x) # Slow stream 
        f_x = self.ConvBlock5(x) # Fast stream 

        # First set of blocks and fusion
        s_x1 = self.Block1(s_x)
        s_x1 = self.MaxpoolSpa(s_x1)
        s_x1 = self.drop_1(s_x1)

        f_x1 = self.Block4(f_x)
        f_x1 = self.MaxpoolSpa(f_x1)
        f_x1 = self.drop_2(f_x1)

        s_x1 = self.fuse_1(s_x1,f_x1) # LateralConnection

        # Second set of blocks and fusion
        s_x2 = self.Block2(s_x1)
        s_x2 = self.MaxpoolSpa(s_x2)
        s_x2 = self.drop_3(s_x2)
        
        f_x2 = self.Block5(f_x1)
        f_x2 = self.MaxpoolSpa(f_x2)
        f_x2 = self.drop_4(f_x2)

        s_x2 = self.fuse_2(s_x2,f_x2) # LateralConnection
        
        # Third blocks and upsampling
        s_x3 = self.Block3(s_x2) 
        s_x3 = self.upsample1(s_x3) 
        s_x3 = self.drop_5(s_x3)

        f_x3 = self.Block6(f_x2)
        f_x3 = self.ConvBlock6(f_x3) 
        f_x3 = self.drop_6(f_x3)

        # Final fusion and upsampling
        x_fusion = torch.cat((f_x3, s_x3), dim=1) 
        x_final = self.upsample2(x_fusion) 

        x_final = self.poolspa(x_final)
        x_final = self.ConvBlockLast(x_final)

        rPPG = x_final.view(-1, length)

        return rPPG    
