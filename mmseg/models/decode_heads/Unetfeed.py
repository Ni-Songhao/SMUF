from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.init as init
from mmcv.cnn import ConvModule
from collections import OrderedDict
import torch.nn.functional as F
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import *
import math
from timm.models.layers import DropPath, trunc_normal_
from mmseg.utils import OptConfigType, SampleList
from typing import Optional, Tuple, Union
from mmseg.models.losses import accuracy
from mmcv.cnn import DepthwiseSeparableConvModule


class SqueezeAndExcitation(nn.Module):
    """Channel squeeze-and-excitation block."""

    def __init__(self, channels, reduction=16, activation=nn.ReLU(inplace=True)):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            activation,
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class SqueezeAndExciteFusionAdd(nn.Module):
    """Apply SE to two branches and fuse them by addition."""

    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.se_branch0 = SqueezeAndExcitation(
            channels_in, activation=activation)
        self.se_branch1 = SqueezeAndExcitation(
            channels_in, activation=activation)

    def forward(self, branch0, branch1):
        return self.se_branch0(branch0) + self.se_branch1(branch1)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2,num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=8):
        super().__init__()
        # assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        # head_dim = dim1 // num_heads

        # self.scale = qk_scale or head_dim ** -0.5

        # self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        # self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim1,
            kdim=dim2,
            vdim=dim2,
            num_heads=num_heads,
            dropout=attn_drop,
            bias = qkv_bias,
            batch_first=True
        )
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim2)
        self.act = nn.GELU()
        # self.kv = nn.Linear(dim2,dim2*2)
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
                
    def _process_single_feature(self, feature, H, W, B, C):
        """处理单个特征"""
        feature = feature.permute(0, 2, 1).reshape(B, C, H, W)
        feature = self.sr(self.pool(feature)).reshape(B, C, -1).permute(0, 2, 1)
        return feature
    
    def forward(self, x, y, H2, W2, H1, W1,modality):
        B1, N1, C1 = x.shape
        x_ = x
        # x_ = x.permute(0, 2, 1).reshape(B1, C1, H1, W1)
        # x_ = self.sr1(self.pool1(x_)).reshape(B1, C1, -1).permute(0, 2, 1)
        # x_ = self.norm1(x_)
        # x_ = self.act(x_)
        # N1 = N1 // (2 * 2)
        # q = self.q(x_)
        if modality == 2:
            y_0, y_1 = y.chunk(2, dim=1)
            B2, N2, C2 = y_0.shape
            # 处理两个模态
            y_0 = self._process_single_feature(y_0, H2, W2, B2, C2)
            y_1 = self._process_single_feature(y_1, H2, W2, B2, C2)
            y_ = torch.cat([y_0, y_1], dim=1)
        else:
            B2, N2, C2 = y.shape
            y_ = self._process_single_feature(y, H2, W2, B2, C2)
        y_ = self.norm(y_)
        y_ = self.act(y_)
        # kv = self.kv(y_)
        # k, v = kv[0], kv[1]
        x, _ = self.mha(x_, y_, y_)
        # x = self.proj(x)
        x = self.proj_drop(x)
        # x = x.transpose(1, 2).view(B1, C1, H1 // 2, W1 // 2)
        # x = resize(x, size=(H1, W1), mode='bilinear', align_corners=False)
        # x = x.flatten(2).transpose(1, 2)

        return x
    

class Block(nn.Module):

    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,pool_ratio=8):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        self.attn = CrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads,pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, y,  H2, W2, H1, W1,modality):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2, H1, W1,modality))
        x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))

        return x
    
    
        
class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8,norm_cfg=dict(type='SynBN')):
        super(WF, self).__init__()
        self.pre_conv = nn.Conv2d(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvModule(
            decode_channels, 
            decode_channels, 
            kernel_size=3, 
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x, res):
        x = F.interpolate(x, size=res.shape[2:], mode='bilinear', align_corners=False)
        weights = F.relu(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * self.pre_conv(x)
        x = self.post_conv(x)
        return x
    
        
class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64,norm_cfg=dict(type='SyncBN')):
        super().__init__()
        self.pre_conv = nn.Conv2d(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvModule(
                                    decode_channels, 
                                    decode_channels, 
                                    kernel_size=3,
                                    padding=1,
                                    norm_cfg=norm_cfg,
                                    act_cfg=dict(type='ReLU')
                                )

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvModule(
                                    decode_channels, 
                                    decode_channels, 
                                    kernel_size=1,
                                    norm_cfg=norm_cfg,
                                    act_cfg=None
                                )
        self.proj = DepthwiseSeparableConvModule(
           in_channels=decode_channels, out_channels=decode_channels,kernel_size=3,padding=1,act_cfg=None,
           dw_norm_cfg = norm_cfg
        )
        self.act = nn.ReLU(inplace=False)

    def forward(self, x, res):
        x = F.interpolate(x, size=res.shape[2:], mode='bilinear', align_corners=False)
        weights = F.relu(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * self.pre_conv(x)
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x.clone()
        ca = self.ca(x) * x.clone()
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x
    
  
@MODELS.register_module()
class UnetFeedHead(BaseDecodeHead):
    """
    尝试统一维度。将所有的维度全部变为64以实现网络模型的缩小。如果效果不行，尝试维度放大
    尤其解码器中使用Transformer架构。
    """
    def __init__(self, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # assert len(feature_strides) == len(self.in_channels)
        
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.attn_c4_c1 = Block(dim1=c4_in_channels, dim2=c1_in_channels, num_heads=8, mlp_ratio=4,
                                drop_path=0.1,pool_ratio=8)
        self.attn_c3_c1 = Block(dim1=c3_in_channels, dim2=c1_in_channels, num_heads=5, mlp_ratio=4,
                                drop_path=0.1,pool_ratio=4)
        self.attn_c2_c1 = Block(dim1=c2_in_channels, dim2=c1_in_channels, num_heads=2, mlp_ratio=4,
                                drop_path=0.1,pool_ratio=2)

        self.linear_fuse = ConvModule(
            in_channels=c1_in_channels + c2_in_channels + c3_in_channels + c4_in_channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )
        # 没有使用feature cross
        self.linear_pred = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        # self.edge_pred = nn.Conv2d(self.channels, 1, kernel_size=1)
        ## expand
        # self.pre_conv = ConvModule(
        #     c4_in_channels,
        #     self.channels,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=None
        # )
        # self.fuse3 = WF(c4_in_channels,c3_in_channels,norm_cfg=self.norm_cfg)
        # self.fuse2 = WF(c3_in_channels,c2_in_channels,norm_cfg=self.norm_cfg)
        # self.FRH = FeatureRefinementHead(c2_in_channels,c1_in_channels,norm_cfg=self.norm_cfg)
        # self.segmentation_head = nn.Sequential(ConvModule(
        #         c1_in_channels,
        #         c1_in_channels,
        #         kernel_size=3,
        #         padding=1,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=dict(type='ReLU')
        #     ),
        #     nn.Dropout2d(p=self.dropout_ratio, inplace=False),
        #     nn.Conv2d(c1_in_channels, self.num_classes, kernel_size=1))
        
        self.se_layer0 = SqueezeAndExciteFusionAdd(c1_in_channels)
        self.se_layer1 = SqueezeAndExciteFusionAdd(c2_in_channels)
        self.se_layer2 = SqueezeAndExciteFusionAdd(c3_in_channels)
        self.se_layer3 = SqueezeAndExciteFusionAdd(c4_in_channels)
        self.linear_2 = nn.Conv2d(c2_in_channels,self.num_classes,kernel_size=1)
        self.linear_3 = nn.Conv2d(c3_in_channels,self.num_classes,kernel_size=1)
        self.linear_4 = nn.Conv2d(c4_in_channels,self.num_classes,kernel_size=1)
        
    def forward(self, inputs):
        # if modality ==2:
        #     x_L,x_S,modality,loss_moe = inputs
        # else:
        x_L,x_S,modality = inputs  # len=4, 1/4,1/8,1/16,1/32
        modality=2  
        c1l, c2l, c3l, c4l = x_L
        c1s, c2s, c3s, c4s = x_S
        ## sum_ feature 有些作用，目前主要的问题是单频下降太厉害了
        c1 = self.se_layer0(c1l,c1s)
        c2 = self.se_layer1(c2l,c2s)
        c3 = self.se_layer2(c3l,c3s)
        c4 = self.se_layer3(c4l,c4s)
        # c1 = c1l+c1s
        # c2 = c2l+c2s
        # c3 = c3s+c3l
        # c4 = c4s+c4l
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape
        c1_kv = torch.cat([c1l.flatten(2).transpose(1, 2),c1s.flatten(2).transpose(1, 2)],dim=1)
        # if modality ==2:
        #     c1_kv = torch.cat([c1l.flatten(2).transpose(1, 2),c1s.flatten(2).transpose(1, 2)],dim=1)
        # else:
        #     c1_kv = c1.flatten(2).transpose(1, 2)
        ############## MLP decoder on C1-C4 ###########
         
        ## stage4 
        # c4 = self.pre_conv(c4)
        _c4 = self.attn_c4_c1(c4.flatten(2).transpose(1, 2), c1_kv, h1,w1, h4,w4,modality)
        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
       
        
        # ## stage3
        # c3 = self.fuse3(_c4,c3)
        _c3 = self.attn_c3_c1(c3.flatten(2).transpose(1, 2), c1_kv, h1,w1, h3,w3,modality)
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        
        # ## stage 2
        # c2 = self.fuse2(_c3,c2)
        _c2 = self.attn_c2_c1(c2.flatten(2).transpose(1, 2), c1_kv, h1,w1,h2, w2,modality)
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        
        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, c1], dim=1))
        # x1 = self.FRH(_c2, c1)
        # x1 = self.segmentation_head(x1)
        x1 = self.dropout(_c)
        x1 = self.linear_pred(x1)
        ## 存在辅助函数和最后输出等问题
        if self.training:
            x4 = self.linear_4(_c4)
            x3 = self.linear_3(_c3)
            x2 = self.linear_2(_c2)
            return x1,x2,x3,x4
        else:
            return x1
        
    # def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
    #     gt_semantic_segs = [
    #         data_sample.gt_sem_seg.data for data_sample in batch_data_samples
    #     ]
    #     gt_edge_segs = [
    #         data_sample.gt_edge_map.data for data_sample in batch_data_samples
    #     ]
    #     gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
    #     gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
    #     return gt_sem_segs, gt_edge_segs
       
    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                        batch_data_samples: SampleList) -> dict:
        loss = dict()
        out_logit, skip1_logit, skip2_logit, skip3_logit = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)

        out_logit = resize(
            out_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        skip1_logit = resize(
            skip1_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        skip2_logit = resize(
            skip2_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        skip3_logit = resize(
            skip3_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)

        loss['loss_out'] = self.loss_decode[0](out_logit, seg_label,ignore_index=255)
        loss['loss_skip1'] = self.loss_decode[1](skip1_logit, seg_label,ignore_index=255)
        loss['loss_skip2'] = self.loss_decode[2](skip2_logit, seg_label,ignore_index=255)
        loss['loss_skip3'] = self.loss_decode[3](skip3_logit, seg_label,ignore_index=255)
        # loss['loss_moe'] = sum(loss_moe) / len(loss_moe)
        loss['acc_seg'] = accuracy(
            out_logit, seg_label, ignore_index=self.ignore_index)

        return loss
