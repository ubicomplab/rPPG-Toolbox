"""
FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing
NeurIPS 2024
Jitesh Joshi, Sos S. Agaian, and Youngjun Cho
"""

import torch
import torch.nn as nn
from neural_methods.model.FactorizePhys.FSAM import FeaturesFactorizationModule

nf = [8, 12, 16]

model_config = {
    "MD_FSAM": True,
    "MD_TYPE": "NMF",
    "MD_TRANSFORM": "T_KAB",
    "MD_R": 1,
    "MD_S": 1,
    "MD_STEPS": 4,
    "MD_INFERENCE": False,
    "MD_RESIDUAL": False,
    "INV_T": 1,
    "ETA": 0.9,
    "RAND_INIT": True,
    "in_channels": 3,
    "data_channels": 4,
    "align_channels": nf[2] // 2,
    "height": 72,
    "weight": 72,
    "batch_size": 4,
    "frames": 160,
    "debug": False,
    "assess_latency": False,
    "num_trials": 20,
    "visualize": False,
    "ckpt_path": "",
    "data_path": "",
    "label_path": ""
}


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=False),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class rPPG_FeatureExtractor(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False):
        super(rPPG_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, 160, 72, 72
        self.FeatureExtractor = nn.Sequential(
            ConvBlock3D(inCh, nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]),  #B, nf[0], 160, 72, 72
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 2, 2], [1, 0, 0]), #B, nf[1], 160, 35, 35
            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[1], 160, 33, 33
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[1], 160, 31, 31
            ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 2, 2], [1, 0, 0]), #B, nf[2], 160, 15, 15
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 13, 13
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        voxel_embeddings = self.FeatureExtractor(x)
        if self.debug:
            print("rPPG Feature Extractor")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)
        return voxel_embeddings


class BVP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(BVP_Head, self).__init__()
        self.debug = debug

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 11, 11
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 9, 9
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

        if self.use_fsam:
            inC = nf[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = nf[2]

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                         #B, nf[1], 160, 5, 5
            ConvBlock3D(nf[1], nf[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                       #B, nf[0], 160, 3, 3
            nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  #B, 1, 160, 1, 1
        )


    def forward(self, voxel_embeddings, batch, length):

        if self.debug:
            print("BVP Head")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        voxel_embeddings = self.conv_block(voxel_embeddings)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(voxel_embeddings - voxel_embeddings.min()) # to make it positive (>= 0)
            else:
                att_mask, appx_error = self.fsam(voxel_embeddings)

            if self.debug:
                print("att_mask.shape", att_mask.shape)

            # # directly use att_mask   ---> difficult to converge without Residual connection. Needs high rank
            # factorized_embeddings = self.fsam_norm(att_mask)

            # # Residual connection: 
            # factorized_embeddings = voxel_embeddings + self.fsam_norm(att_mask)

            if self.md_res:
                # Multiplication with Residual connection
                x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)
                factorized_embeddings = voxel_embeddings + factorized_embeddings
            else:
                # Multiplication
                x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)            

            # # Concatenate
            # factorized_embeddings = torch.cat([voxel_embeddings, self.fsam_norm(x)], dim=1)

            x = self.final_layer(factorized_embeddings)
        
        else:
            x = self.final_layer(voxel_embeddings)

        rPPG = x.view(-1, length)

        if self.debug:
            print("     rPPG.shape", rPPG.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rPPG, factorized_embeddings, appx_error
        else:
            return rPPG



class FactorizePhys(nn.Module):
    def __init__(self, frames, md_config, in_channels=3, dropout=0.1, device=torch.device("cpu"), debug=False):
        super(FactorizePhys, self).__init__()
        self.debug = debug

        self.in_channels = in_channels
        if self.in_channels == 1 or self.in_channels == 3:
            self.norm = nn.InstanceNorm3d(self.in_channels)
        elif self.in_channels == 4:
            self.rgb_norm = nn.InstanceNorm3d(3)
            self.thermal_norm = nn.InstanceNorm3d(1)
        else:
            print("Unsupported input channels")
        
        self.use_fsam = md_config["MD_FSAM"]
        self.md_infer = md_config["MD_INFERENCE"]

        for key in model_config:
            if key not in md_config:
                md_config[key] = model_config[key]

        if self.debug:
            print("nf:", nf)

        self.rppg_feature_extractor = rPPG_FeatureExtractor(self.in_channels, dropout_rate=dropout, debug=debug)

        self.rppg_head = BVP_Head(md_config, device=device, dropout_rate=dropout, debug=debug)

        
    def forward(self, x): # [batch, Features=3, Temp=frames, Width=32, Height=32]
        
        [batch, channel, length, width, height] = x.shape
        
        # if self.in_channels == 1:
        #     x = x[:, :, :-1, :, :]
        # else:
        #     x = torch.diff(x, dim=2)
        
        x = torch.diff(x, dim=2)

        if self.debug:
            print("Input.shape", x.shape)

        if self.in_channels == 1:
            x = self.norm(x[:, -1:, :, :, :])
        elif self.in_channels == 3:
            x = self.norm(x[:, :3, :, :, :])
        elif self.in_channels == 4:
            rgb_x = self.rgb_norm(x[:, :3, :, :, :])
            thermal_x = self.thermal_norm(x[:, -1:, :, :, :])
            x = torch.concat([rgb_x, thermal_x], dim = 1)
        else:
            try:
                print("Specified input channels:", self.in_channels)
                print("Data channels", channel)
                assert self.in_channels <= channel
            except:
                print("Incorrectly preprocessed data provided as input. Number of channels exceed the specified or default channels")
                print("Default or specified channels:", self.in_channels)
                print("Data channels [B, C, N, W, H]", x.shape)
                print("Exiting")
                exit()

        if self.debug:
            print("Diff Normalized shape", x.shape)

        voxel_embeddings = self.rppg_feature_extractor(x)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            rPPG, factorized_embeddings, appx_error = self.rppg_head(voxel_embeddings, batch, length-1)
        else:
            rPPG = self.rppg_head(voxel_embeddings, batch, length-1)

        # if self.debug:
        #     print("rppg_feats.shape", rppg_feats.shape)

        # rPPG = rppg_feats.view(-1, length-1)

        if self.debug:
            print("rPPG.shape", rPPG.shape)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rPPG, voxel_embeddings, factorized_embeddings, appx_error
        else:
            return rPPG, voxel_embeddings