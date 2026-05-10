from .extra_module.MoE_share import MoE

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.model.weight_init import (constant_init, normal_init,
                                      trunc_normal_init)
from mmseg.registry import MODELS
from mmcv.cnn import build_norm_layer
import math
from functools import partial
import warnings
from ..utils import resize
import torch.nn.functional as F
from torch.nn.init import normal_


class InterleavedGroupConv2d(nn.Module):
    """Interleaved group convolution used by the LSKNet patch embedding."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 primary_partition=1,
                 secondary_partition=1,
                 norm_cfg=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=primary_partition,
            bias=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            1,
            1,
            0,
            1,
            groups=secondary_partition,
            bias=False)
        self.primary_partition = primary_partition
        self.secondary_partition = secondary_partition
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        else:
            self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.reorder(x, self.primary_partition)
        x = self.conv2(x)
        x = self.reorder(x, self.secondary_partition)
        _, _, h, w = x.size()
        return x, h, w

    def reorder(self, x, branch_factor):
        n, c, h, w = x.size()
        x = x.view(n, branch_factor, c // branch_factor, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(n, c, h, w)
        return self.norm(x)


class ADA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0_0 = nn.Conv2d(dim, 8, 1)
        self.conv0_1 =nn.Conv2d(dim, 8, 1)
        self.conv = nn.Conv2d(8, dim, 1)

    def forward(self, p, x):
        p = self.conv0_0(p)
        x = self.conv0_1(x)
        p1 = p*x
        p1 = self.conv(p1)
        return p1

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 moe=False,
                 mlp_ratio=4.,
                 reduction=8,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=None,
                 moe_num_experts=4,
                 moe_topk=2):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path)) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        if moe:
            self.moe = MoE(
                input_size=dim,
                output_size=dim,
                num_experts=int(moe_num_experts),
                hidden_size=64,
                shared_hidden=mlp_hidden_dim,
                k=int(moe_topk))
        else:
            self.moe = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        # y,loss = self.moe(self.norm2(x))
        short_cut = x.clone()
        y = self.moe(self.norm2(x))
        # loss=0
        x = short_cut + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * y)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W


@MODELS.register_module()
class LSKNet_Base(BaseModule):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],probs=[1,1,1],
                mlp_ratios=[8,8,4,4], reduction_ratio=[4,8,20,32],drop_rate=0., drop_path_rate=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, 
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages
        self.probabilities = probs

        # self.prompt =  nn.Parameter(torch.randn(1,64, 32, 32))
        # self.linear_2 = nn.Conv2d(embed_dims[0],embed_dims[1],kernel_size=1)
        # self.linear_3 = nn.Conv2d(embed_dims[0],embed_dims[2],kernel_size=1)
        # self.linear_4 = nn.Conv2d(embed_dims[0],embed_dims[3],kernel_size=1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.prompt = nn.Parameter(torch.zeros(2,embed_dims[0], 48, 48))
        # self.prompt = torch.zeros(2,embed_dims[0],48,48).cuda()
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            # ada = ADA(embed_dims[i])
            
            block = nn.ModuleList([Block(
                dim=embed_dims[i], moe = True if j == depths[i]-1 else False,
                mlp_ratio=mlp_ratios[i], reduction=reduction_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j],norm_cfg=norm_cfg)
                for j in range(depths[i])])
            # block = nn.ModuleList([Block(
            #     dim=embed_dims[i], moe=False,
            #     mlp_ratio=mlp_ratios[i], reduction=reduction_ratio[i],
            #     drop=drop_rate, drop_path=dpr[cur + j], norm_cfg=norm_cfg)
            #     for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            linear = nn.Conv2d(in_channels=embed_dims[0],
                               out_channels = embed_dims[i],
                               kernel_size=1)
            # setattr(self, f"ada{i + 1}", ada)
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"linear{i + 1}", linear)

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()
        
    def forward_stage(self, x,H,W,i):
        B = x.shape[0]
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        # ada = getattr(self, f"ada{i + 1}")
        # prompted = ada(x,prompt)
        # x = x+prompted
        for blk in block:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)
        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
                # ,loss_moe)
    def generation_index(self):
        probabilities = torch.tensor(self.probabilities, dtype=torch.float)
        random_numbers = torch.multinomial(probabilities, num_samples=1, replacement=True)
        return random_numbers.item()
    
    def forward(self, x,modality,mask):
        # loss = []
        x_L = []
        x_S = []
        if self.training:
            modality = self.generation_index()
        else:
            modality = modality
            # modality =0
        # dim = x.shape[1]
        # x[:,:dim//2,:,:]=0
        x1,x2 = x.chunk(2, dim=1)
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            x1, H, W = patch_embed(x1)
            x2,_,_ = patch_embed(x2)
            if i==0:
                x1 = x1 + resize(self.prompt[0].expand(x1.size(0), -1, -1, -1), size=(H, W), mode='bilinear', align_corners=False)
                x2 = x2 + resize(self.prompt[1].expand(x1.size(0), -1, -1, -1), size=(H, W), mode='bilinear', align_corners=False)
            x1 = self.forward_stage(x1,H,W,i)
            x2 = self.forward_stage(x2,H,W,i)
            # loss.append(loss1+loss2)
            if modality == 2:
                x_L.append(x1)
                x_S.append(x2)
            elif modality == 0:
                linear = getattr(self, f"linear{i + 1}")
                x_prompt_L = resize(linear(self.prompt[0].expand(x1.size(0), -1, -1, -1)), size=(H, W), mode='bilinear', align_corners=False)
                x_L.append(x1)
                x_S.append(x_prompt_L)
            elif modality == 1:
                linear = getattr(self, f"linear{i + 1}")
                x_prompt_S = resize(linear(self.prompt[1].expand(x1.size(0), -1, -1, -1)), size=(H, W), mode='bilinear', align_corners=False)
                x_L.append(x_prompt_S)
                x_S.append(x2)

        return x_L,x_S,modality


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
