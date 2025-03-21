""" 
RhythmFormer:Extracting rPPG Signals Based on Hierarchical Temporal Periodic Transformer
"""
from typing import Optional
import torch
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F
import math
from typing import Tuple, Union
from timm.models.layers import trunc_normal_, DropPath



"""
Adapted from here: https://github.com/rayleizhu/BiFormer
"""
import torch
from torch import Tensor, LongTensor , nn
import torch.nn.functional as F
from typing import Optional, Tuple
            
def _grid2seq(x:Tensor, region_size:Tuple[int], num_heads:int):
    """
    Args:
        x: BCTHW tensor
        region size: int
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_t, region_h, region_w: number of regions per t/col/row
    """
    B, C, T, H, W = x.size()
    region_t ,region_h, region_w = T//region_size[0],  H//region_size[1],  W//region_size[2]
    x = x.view(B, num_heads, C//num_heads, region_t, region_size[0],region_h, region_size[1], region_w, region_size[2])
    x = torch.einsum('bmdtohpwq->bmthwopqd', x).flatten(2, 4).flatten(-4, -2) # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_t, region_h, region_w


def _seq2grid(x:Tensor, region_t:int, region_h:int, region_w:int, region_size:Tuple[int]):
    """
    Args: 
        x: (bs, nhead, nregion, reg_size^2, head_dim)
    Return:
        x: (bs, C, T, H, W)
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_t, region_h, region_w, region_size[0], region_size[1], region_size[2], head_dim)
    x = torch.einsum('bmthwopqd->bmdtohpwq', x).reshape(bs, nhead*head_dim,
        region_t*region_size[0],region_h*region_size[1], region_w*region_size[2])
    return x


def video_regional_routing_attention_torch(
    query:Tensor, key:Tensor, value:Tensor, scale:float,
    region_graph:LongTensor, region_size:Tuple[int],
    kv_region_size:Optional[Tuple[int]]=None,
    auto_pad=False)->Tensor:
    """
    Args:
        query, key, value: (B, C, T, H, W) tensor
        scale: the scale/temperature for dot product attention
        region_graph: (B, nhead, t_q*h_q*w_q, topk) tensor, topk <= t_k*h_k*w_k
        region_size: region/window size for queries, (rt, rh, rw)
        key_region_size: optional, if None, key_region_size=region_size
    Return:
        output: (B, C, T, H, W) tensor
        attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()
    
    # # Auto pad to deal with any input size 
    # q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    # if auto_pad:
    #     _, _, Hq, Wq = query.size()
    #     q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
    #     q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
    #     if (q_pad_b > 0 or q_pad_r > 0):
    #         query = F.pad(query, (0, q_pad_r, 0, q_pad_b)) # zero padding

    #     _, _, Hk, Wk = key.size()
    #     kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
    #     kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
    #     if (kv_pad_r > 0 or kv_pad_b > 0):
    #         key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b)) # zero padding
    #         value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b)) # zero padding
    
    # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
    query, q_region_t, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # gather key and values.
    # torch.gather does not support broadcasting, hence we do it manually
    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1).\
        expand(-1, -1, -1, -1, kv_region_size, head_dim)
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).\
        expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
        index=broadcasted_region_graph) # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).\
        expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
        index=broadcasted_region_graph) # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    
    # token-to-token attention
    # (bs, nhead, q_nregion, reg_size, head_dim) @ (bs, nhead, q_nregion, head_dim, topk*kv_region_size)
    # -> (bs, nhead, q_nregion, reg_size, topk*kv_region_size)
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    # (bs, nhead, q_nregion, reg_size, topk*kv_region_size) @ (bs, nhead, q_nregion, topk*kv_region_size, head_dim)
    # -> (bs, nhead, q_nregion, reg_size, head_dim)
    output = attn @ value_g.flatten(-3, -2)

    # to BCTHW format
    output = _seq2grid(output, region_t=q_region_t, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    # remove paddings if needed
    # if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
    #     output = output[:, :, :Hq, :Wq]

    return output, attn




class CDC_T(nn.Module):
    """
    The CDC_T Module is from here: https://github.com/ZitongYu/PhysFormer/model/transformer_layer.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
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
      
class video_BRA(nn.Module):

    def __init__(self, dim, num_heads=8, t_patch=8, qk_scale=None, topk=4,  side_dwconv=3, auto_pad=False, attn_backend='torch'):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5 
        self.topk = topk
        self.t_patch = t_patch  # frame of patch
        ################side_dwconv (i.e. LCE in Shunted Transformer)###########
        self.lepe = nn.Conv3d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        ##########################################
        self.qkv_linear = nn.Conv3d(self.dim, 3*self.dim, kernel_size=1)
        self.output_linear = nn.Conv3d(self.dim, self.dim, kernel_size=1)
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=0.2),  
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=0.2),  
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        if attn_backend == 'torch':
            self.attn_fn = video_regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x:Tensor):

        N, C, T, H, W = x.size()
        t_region = max(4 // self.t_patch , 1)
        region_size = (t_region, H//4 , W//4)

        # STEP 1: linear projection
        q , k , v = self.proj_q(x) , self.proj_k(x) ,self.proj_v(x)

        # STEP 2: pre attention
        q_r = F.avg_pool3d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool3d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False) # ncthw
        q_r:Tensor = q_r.permute(0, 2, 3, 4, 1).flatten(1, 3) # n(thw)c
        k_r:Tensor = k_r.flatten(2, 4) # nc(thw)
        a_r = q_r @ k_r # n(thw)(thw)
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1) # n(thw)k
        idx_r:LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1) 

        # STEP 3: refined attention
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size)
        
        output = output + self.lepe(v) # nctHW
        output = self.output_linear(output) # nctHW

        return output

class video_BiFormerBlock(nn.Module):
    def __init__(self, dim, drop_path=0., num_heads=4, t_patch=1,qk_scale=None, topk=4, mlp_ratio=2, side_dwconv=5):
        super().__init__()
        self.t_patch = t_patch
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = video_BRA(dim=dim, num_heads=num_heads, t_patch=t_patch,qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv)
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = nn.Sequential(nn.Conv3d(dim, int(mlp_ratio*dim), kernel_size=1),
                                 nn.BatchNorm3d(int(mlp_ratio*dim)),
                                 nn.GELU(),
                                 nn.Conv3d(int(mlp_ratio*dim),  int(mlp_ratio*dim), 3, stride=1, padding=1),  
                                 nn.BatchNorm3d(int(mlp_ratio*dim)),
                                 nn.GELU(),
                                 nn.Conv3d(int(mlp_ratio*dim), dim, kernel_size=1),
                                 nn.BatchNorm3d(dim),
                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Fusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5):
        super(Fusion_Stem, self).__init__()

        self.stem11 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/4,W/4]
        """
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x2 = torch.cat([x[:,:1,:,:,:],x[:,:D-1,:,:,:]],1)
        x3 = x
        x4 = torch.cat([x[:,1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)
        x = self.apha*x_path1 + self.belta*x_path2
        
        return x
    
class TPT_Block(nn.Module):
    def __init__(self, dim, depth, num_heads, t_patch, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=5):
        super().__init__()
        self.dim = dim
        self.depth = depth
        ############ downsample layers & upsample layers #####################
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.layer_n = int(math.log(t_patch,2))
        for i in range(self.layer_n):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(dim), 
                nn.Conv3d(dim , dim , kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                )
            self.downsample_layers.append(downsample_layer)
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=(2, 1, 1)),
                nn.Conv3d(dim , dim , [3, 1, 1], stride=1, padding=(1, 0, 0)),   
                nn.BatchNorm3d(dim),
                nn.ELU(),
                )
            self.upsample_layers.append(upsample_layer)
        ######################################################################
        self.blocks = nn.ModuleList([
            video_BiFormerBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    num_heads=num_heads,
                    t_patch=t_patch,
                    topk=topk,
                    mlp_ratio=mlp_ratio,
                    side_dwconv=side_dwconv,
                )
            for i in range(depth)
        ])
    def forward(self, x:torch.Tensor):
        """Definition of TPT_Block.
        Args:
          x [N,C,D,H,W]
        Returns:
          x [N,C,D,H,W]
        """
        for i in range(self.layer_n) :
            x = self.downsample_layers[i](x)
        for blk in self.blocks:
            x = blk(x)
        for i in range(self.layer_n) :
            x = self.upsample_layers[i](x)

        return x
    
class RhythmFormer(nn.Module):

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        dim: int = 64, frame: int = 160,
        image_size: Optional[int] = (160,128,128),
        in_chans=64, head_dim=16,
        stage_n = 3,
        embed_dim=[64, 64, 64], mlp_ratios=[1.5, 1.5, 1.5],
        depth=[2, 2, 2], 
        t_patchs:Union[int, Tuple[int]]=(2, 4, 8),
        topks:Union[int, Tuple[int]]=(40, 40, 40),
        side_dwconv:int=3,
        drop_path_rate=0.,
        use_checkpoint_stages=[],
    ):
        super().__init__()

        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim              
        self.stage_n = stage_n

        self.Fusion_Stem = Fusion_Stem()
        self.patch_embedding = nn.Conv3d(in_chans,embed_dim[0], kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.ConvBlockLast = nn.Conv1d(embed_dim[-1], 1, kernel_size=1,stride=1, padding=0)

        ##########################################################################
        self.stages = nn.ModuleList()
        nheads= [dim // head_dim for dim in embed_dim]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        for i in range(stage_n):
            stage = TPT_Block(dim=embed_dim[i],
                               depth=depth[i],
                               num_heads=nheads[i], 
                               mlp_ratio=mlp_ratios[i],
                               drop_path=dp_rates[sum(depth[:i]):sum(depth[:i+1])],
                               t_patch=t_patchs[i], topk=topks[i], side_dwconv=side_dwconv
                               )
            self.stages.append(stage)
        ##########################################################################

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        N, D, C, H, W = x.shape
        x = self.Fusion_Stem(x)    #[N*D 64 H/4 W/4]
        x = x.view(N,D,64,H//4,W//4).permute(0,2,1,3,4)
        x = self.patch_embedding(x)    #[N 64 D 8 8]
        for i in range(3):
            x = self.stages[i](x)    #[N 64 D 8 8]
        features_last = torch.mean(x,3)    #[N, 64, D, 8]  
        features_last = torch.mean(features_last,3)    #[N, 64, D]  
        rPPG = self.ConvBlockLast(features_last)    #[N, 1, D]
        rPPG = rPPG.squeeze(1)
        return rPPG 
