import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm

from .select_feature_dual import BiAttentionBlock
from .swin_transformer_dual import SwinTransformer
from .swin_transformer_unet_skip_expand_decoder_sys_dual_heat_map import SwinTransformerSysfinal_decoder, SwinTransformerSys
import copy

from .net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction, TransformerBlock

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

# class Conv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
#         super(Conv, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
#         )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=bias,
                      stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=bias,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out

class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x

class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6
                 ):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        # # build layers
        # self.layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
        #                         input_resolution=(patches_resolution[0] // (2 ** i_layer),
        #                                             patches_resolution[1] // (2 ** i_layer)),
        #                         depth=depths[i_layer],
        #                         num_heads=num_heads[i_layer],
        #                         window_size=window_size,
        #                         mlp_ratio=self.mlp_ratio,
        #                         qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                         drop=drop_rate, attn_drop=attn_drop_rate,
        #                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #                         norm_layer=norm_layer,
        #                         downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
        #                         use_checkpoint=use_checkpoint,
        #                         fused_window_process=fused_window_process)
        #     self.layers.append(layer)
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            
            # print("res4shape",res4.shape) [8, 512, 16, 16] 
            # print("self.pre_conv(res4)",self.pre_conv(res4).shape)  [8, 64, 16, 16] 
            # print("self.b4(self.pre_conv(res4))",self.b4(self.pre_conv(res4)).shape) [8, 64, 16, 16]

            x = self.b4(self.pre_conv(res4))
            h4 = self.up4(x)

            x = self.p3(x, res3) # 
            # print("x",x.shape)
            x = self.b3(x)
            h3 = self.up3(x)

            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            x = self.p1(x, res1)
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)

            return x, ah
        else:
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            x = self.b3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            x = self.p1(x, res1)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

# class 
# class UNetFormer(nn.Module):
#     def __init__(self,
#                  decode_channels=64,
#                  dropout=0.1,
#                  backbone_name='resnet50',
#                  pretrained=True,
#                  window_size=8,
#                  num_classes=6,
#                  backbone_channels = [256, 512, 1024, 2048],
#                  out_channels = [96, 192, 384, 768]
#                  ):
#         super().__init__()

#         self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
#                                           out_indices=(1, 2, 3, 4), pretrained=pretrained)
#         self.backboned = timm.create_model(backbone_name, features_only=True, output_stride=32,
#                                           out_indices=(1, 2, 3, 4), pretrained=pretrained)
        
#         self.layers_convbn = nn.ModuleList()
#         self.layers_convbn_len = len(backbone_channels)
#         for i in range(self.layers_convbn_len):
#             self.layers_convbn.append(ConvBN(backbone_channels[i], out_channels[i], kernel_size=1, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False))
        
#         self.layers_convbn_d = nn.ModuleList()
#         self.layers_convbn_len_d = len(backbone_channels)
#         for i in range(self.layers_convbn_len_d):
#             self.layers_convbn_d.append(ConvBN(backbone_channels[i], out_channels[i], kernel_size=1, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False))

#         encoder_channels = self.backbone.feature_info.channels()

#         self.decoder = SwinTransformerSys(num_classes=7)
#         # self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)


#     def forward(self, x, y):
#         h, w = x.size()[-2:]
#         res = self.backbone(x)
#         resd = self.backboned(y)
    
#         if self.training:
#             x_downsample = []
#             for i in range(len(res)):
#                 x = self.layers_convbn[i](res[i])
#                 x = x.flatten(2).permute(0, 2, 1)
#                 x_downsample.append(x)

#             y_downsample_d = []
#             for i in range(len(resd)):
#                 y = self.layers_convbn_d[i](resd[i])
#                 y = y.flatten(2).permute(0, 2, 1)
#                 y_downsample_d.append(y)
#             result, resultd = self.decoder(x, x_downsample,y, y_downsample_d)
#             return res, resd, result, resultd
#         else:
#             return res, resd
#         return result, resultd

# 定义上采样层
def create_upsample_layer(in_channels, out_channels, scale_factor=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor, padding=scale_factor // 2, output_padding=scale_factor % 2),
        # nn.Conv2d(out_channels, out_channels, kernel_size=3,
        #               stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# 定义上采样层
def upsample(tensor, target_size):
    return F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)

def _get_clones(module, N, layer_share=False):
    # import ipdb; ipdb.set_trace()
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# import matplotlib
# matplotlib.use('Agg')  # 在导入plt之前设置后端
# import matplotlib.pyplot as plt
# import os
# import os
# import torch
# from PIL import Image
# import numpy as np

class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet50',
                #  backbone_name='swsl_resnet18',
                 img_dim = 256,
                 pretrained=True,
                 select_num = 3, 
                 window_size=8,
                 num_classes=6,
                 num_blocks = [4, 4],
                 heads=[8, 8, 8],
                 backbone_channels = [256, 512, 1024, 2048],
                #  backbone_channels = [64, 128, 256, 512],
                 out_channels = [96, 192, 384, 768],
                 LayerNorm_type ='WithBias',
                 bias = False,
                 ffn_expansion_factor = 2,
                 dim = 256
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        self.backboned = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)

        self.backboned.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.layers_convbn = nn.ModuleList()
        self.layers_convbn_len = len(backbone_channels)
        for i in range(self.layers_convbn_len):
            self.layers_convbn.append(ConvBN(backbone_channels[i], out_channels[i], kernel_size=1, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False))
        
        self.layers_convbn_d = nn.ModuleList()
        self.layers_convbn_len_d = len(backbone_channels)
        for i in range(self.layers_convbn_len_d):
            self.layers_convbn_d.append(ConvBN(backbone_channels[i], out_channels[i], kernel_size=1, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False))
    
        
        self.local_feature_convbn = nn.ModuleList()
        self.local_feature_convbn_len = len(backbone_channels)
        for i in range(self.local_feature_convbn_len):
            self.local_feature_convbn.append(ConvBN(backbone_channels[i], img_dim, kernel_size=1, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False))
        
        self.local_feature_convbnd = nn.ModuleList()
        self.local_feature_convbnd_len = len(backbone_channels)
        for i in range(self.local_feature_convbnd_len):
            self.local_feature_convbnd.append(ConvBN(backbone_channels[i], img_dim, kernel_size=1, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False))

        # self.select_feature_layer = BiAttentionBlock(img_dim=256, d_dim=256, embed_dim=1024, num_heads=4)  # 1024
        # self.select_feature_layer = BiAttentionBlock(img_dim=256, d_dim=256, embed_dim=768, num_heads=8)
        self.select_feature_layer = BiAttentionBlock(img_dim=256, d_dim=256, embed_dim=1024, num_heads=8)  # 1024

        self.select_layers = _get_clones(self.select_feature_layer, select_num)

        self.select_layers_split = _get_clones(self.select_feature_layer, select_num)


        self.conv_downsample = Conv(256,256)
        self.conv_downsample_layers = _get_clones(self.conv_downsample, select_num)

        self.features_proj = nn.Linear(img_dim*2, img_dim)
        self.features_proj_layers = _get_clones(self.features_proj, select_num)

        # self._proj = nn.Linear(3840, 96)
        # self.feature_merged_proj = nn.Linear(1024, 96)
        # self.select_feature_merged_proj = nn.Linear(1024, 96)
   
        # # 创建上采样层实例
        self.encode_sw_proj_layers = nn.ModuleList([
            nn.Linear(256, 96),
            nn.Linear(256, 192),
            nn.Linear(256, 384),
            nn.Linear(256, 768)
        ])

        # self.d_select_proj_layers = nn.ModuleList([
        #     nn.Linear(256, 96),
        #     nn.Linear(256, 192),
        #     nn.Linear(256, 384),
        #     nn.Linear(256, 768)
        # ])

        # self.d_select_query_proj_layers = nn.ModuleList([
        #     nn.Linear(4096, 1024),
        #     nn.Linear(1024, 256),
        #     nn.Linear(256, 64),
        #     nn.Linear(256, 768)
        # ])

        self.encoder = SwinTransformer()

        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = SwinTransformerSys(num_classes=num_classes)
        self.final_decoder = SwinTransformerSysfinal_decoder(num_classes=num_classes)

        # pool_scales = self.generate_arithmetic_sequence(1, 64, 64 // 8)
        # self.pool_len = len(pool_scales)
        # self.pool_layers = nn.ModuleList()
        # self.pool_layers.append(nn.Sequential(
        #             ConvBNReLU(1024, 128, kernel_size=1),
        #             nn.AdaptiveAvgPool2d(1)
        #             ))
        # for pool_scale in pool_scales[1:]:
        #     self.pool_layers.append(
        #         nn.Sequential(
        #             nn.AdaptiveAvgPool2d(pool_scale),
        #             ConvBNReLU(1024, 128, kernel_size=1)
        #             ))
        
        # self.shift_dim = ConvBNReLU(2048, 1024, kernel_size=1)

        self.BaseFeature = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.DetailFeature = DetailFeatureExtraction()
        self.encoder_global = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.BaseFeature_d = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.DetailFeature_d = DetailFeatureExtraction()
        self.encoder_global_d = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.res_feature_aggregation = nn.Sequential(*[ConvBN(1024, 256), ConvBN(256, 256), ConvBN(256, 256)])
        self.feature_merged_proj = ConvBN(512,96)
        self.select_d_proj = nn.ModuleList([
            nn.Linear(256, 192),
            nn.Linear(256, 384),
            nn.Linear(256, 768)
        ])
        self.readuce_proj = ConvBN(1024,256)
        self.readuce_d_proj = ConvBN(1024,256)
        self.d_select_proj = ConvBN(768,256)
        self.select_all_feature_proj = ConvBN(1024,256)


    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence

    def get_local_features_embed(self, features, trans_layers):
        trans_features = []
        features_same_dim = []
        for i, f in enumerate(features):
            f = trans_layers[i](f)
            features_same_dim.append(f)
            f = f.flatten(2).permute(0, 2, 1)
            trans_features.append(f)
        return trans_features, features_same_dim

    def get_img_query(self, features):
        querys = []
        for i, f in enumerate(features):
            if i == 0:
                querys.append(f.flatten(2).permute(0,2,1))
            else:
                # print("f------", f.shape)
                # print("features[i-1]", features[i-1].shape)
                # print("self.conv_downsample_layers[i](features[i-1])",self.conv_downsample_layers[i-1](features[i-1]).shape)
                f = torch.cat((f,self.conv_downsample_layers[i-1](features[i-1])), dim=1)
                f = self.features_proj_layers[i-1](f.flatten(2).permute(0,2,1))
                querys.append(f)
        return querys

    # def get_img_query(self, features):
    #     querys = []
    #     for i, f in enumerate(features):
    #             querys.append(f.flatten(2).permute(0,2,1))
    #     return querys

    def forward(self, x, y):
        h, w = x.size()[-2:]
        res = self.backbone(x)
        resd = self.backboned(y)

        x_local_features_embed, x_local_features_same_dim = self.get_local_features_embed(res, self.local_feature_convbn)
        y_local_features_embed, y_local_features_same_dim = self.get_local_features_embed(resd, self.local_feature_convbnd)
        img_querys = self.get_img_query(x_local_features_same_dim)
        for img_querydd in img_querys: 
            print("img_querys",img_querydd.shape)

        res_feature = [upsample(f, (64, 64)) for f in x_local_features_same_dim]
        res_feature = torch.cat(res_feature, dim=1)
        res_feature = self.res_feature_aggregation(res_feature)

        x_local_features_up_sampled = [upsample(f, (32, 32)) for f in x_local_features_same_dim]
        x_local_features_merged = torch.cat(x_local_features_up_sampled, dim=1)
        x_local_features_merged = self.readuce_proj(x_local_features_merged)
        
        y_local_features_up_sampled = [upsample(f, (32, 32)) for f in y_local_features_same_dim]
        y_local_features_merged = torch.cat(y_local_features_up_sampled, dim=1)
        y_local_features_merged = self.readuce_d_proj(y_local_features_merged)
        
        # 频域全局和局部
        x_global_features_merged = self.encoder_global(x_local_features_merged) # (B. 256, 32, 32)
        y_global_features_merged = self.encoder_global_d(y_local_features_merged)
        basefeature = self.BaseFeature(x_global_features_merged) 
        detailfeature = self.DetailFeature(x_global_features_merged)
        basefeature_d = self.BaseFeature_d(y_global_features_merged)
        detailfeature_d = self.DetailFeature_d(y_global_features_merged)
        slect_basefeature_d = self.select_layers_split[0](basefeature.flatten(2).permute(0,2,1), basefeature_d.flatten(2).permute(0,2,1))[0]
        # print("slect_basefeature_d",slect_basefeature_d.shape)
        slect_basefeature_d = slect_basefeature_d.permute(0,2,1)
        s_b, s_n, s_d = slect_basefeature_d.shape
        slect_basefeature_d = slect_basefeature_d.view(s_b, s_n, int(s_d**0.5), int(s_d**0.5))
        # print("slect_basefeature_d",slect_basefeature_d.shape)
        slect_detailfeature_d = self.select_layers_split[1](detailfeature.flatten(2).permute(0,2,1), detailfeature_d.flatten(2).permute(0,2,1))[0]
        # print("slect_detailfeature_d",slect_detailfeature_d.shape)
        slect_detailfeature_d = slect_detailfeature_d.permute(0,2,1)
        s_b, s_n, s_d = slect_detailfeature_d.shape
        slect_detailfeature_d = slect_detailfeature_d.view(s_b, s_n, int(s_d**0.5), int(s_d**0.5))
        # print("slect_detailfeature_d",slect_detailfeature_d.shape)


        # # 创建一个目录来保存热力图
        # output_dir = '/opt/data/private/xffproject/our_method/heatmaps'
        # os.makedirs(output_dir, exist_ok=True)

        # # 遍历每个样本
        # output_reshaped = slect_basefeature_d
        # for sample_index in range(output_reshaped.size(0)):  # 10个样本
        #     sample = output_reshaped[sample_index]
            
        #     # 对于每个样本，遍历所有通道
        #     for channel_index in range(sample.size(0)):  # 256个通道
        #         # 获取特定通道的数据，并转换为numpy数组
        #         heatmap_data = sample[channel_index].detach().cpu().numpy()
                
        #         # 保存热力图为文件，不进行显示
        #         filename = os.path.join(output_dir, f'sample_{sample_index + 1}_channel_{channel_index + 1}.png')
        #         plt.imsave(filename, heatmap_data)
                
        # print("所有热力图已保存至", output_dir)

        # for sample_index in range(output_reshaped.size(0)):  # 10个样本
        #     sample = output_reshaped[sample_index]
            
        #     # 对于每个样本，遍历所有通道
        #     for channel_index in range(sample.size(0)):  # 256个通道
        #         # 获取特定通道的数据，并转换为numpy数组
        #         grayscale_data = sample[channel_index].detach().cpu().numpy()
                
        #         # 将灰度图数据归一化到0-255范围，并转为8位无符号整数
        #         grayscale_normalized = ((grayscale_data - grayscale_data.min()) * (255.0 / (grayscale_data.max() - grayscale_data.min()))).astype(np.uint8)
                
        #         # 使用PIL.Image将numpy数组转换为图像
        #         grayscale_image = Image.fromarray(grayscale_normalized, mode='L')  # 'L'模式代表灰度图像
                
        #         # 保存灰度图为文件
        #         filename = os.path.join(output_dir, f'sample_{sample_index + 1}_channel_{channel_index + 1}.png')
        #         grayscale_image.save(filename)

        # 空间域全局
        slect_global_feature = self.select_layers_split[2](x_global_features_merged.flatten(2).permute(0,2,1), y_global_features_merged.flatten(2).permute(0,2,1))[0] # (B,1024,256)
        # print("slect_global_feature", slect_global_feature.shape)
        slect_global_feature = slect_global_feature.permute(0,2,1)
        s_b, s_n, s_d = slect_global_feature.shape
        slect_global_feature = slect_global_feature.view(s_b, s_n, int(s_d**0.5), int(s_d**0.5))
        
        # x_local_features_merged = x_local_features_merged.flatten(2).permute(0, 2, 1)
        # x_local_features_merged = self.feature_merged_proj(x_local_features_merged)



        x_new = []
        for i, features_embed in enumerate(x_local_features_embed):
            if self.encode_sw_proj_layers[i] is not None:
                features_embed = self.encode_sw_proj_layers[i](features_embed)
            x_new.append(features_embed)
            # d_select torch.Size([10, 4096, 256])
            # d_select torch.Size([10, 1024, 256])
            # d_select torch.Size([10, 256, 256])
            # d_select torch.Size([10, 64, 256])
        
        # 空间域局部
        select_d_features = []
        select_attens = []
        select_original = []
        for i, select in enumerate(self.select_layers):
            d_select, atten_img2d = select(img_querys[i+1], y_local_features_embed[i+1])
            select_original.append(d_select)
            d_select = d_select.permute(0,2,1)
            s_b, s_n, s_d = d_select.shape
            d_select = d_select.view(s_b, s_n, int(s_d**0.5), int(s_d**0.5))
            # print("d_select", d_select.shape)
            if s_d**0.5 != 32:
                d_select = upsample(d_select, (32, 32))
            select_d_features.append(d_select)
            select_attens.append(atten_img2d)
        d_select = torch.cat(select_d_features, dim=1) # (B, 768, 32,32)
        d_select = self.d_select_proj(d_select)
        

        # print("slect_basefeature_d",slect_basefeature_d.shape)
        # print("slect_detailfeature_d",slect_detailfeature_d.shape)
        # print("slect_global_feature",slect_global_feature.shape)
        # print("d_select",d_select.shape)
        slect_all_feature = [slect_basefeature_d, slect_detailfeature_d, slect_global_feature, d_select]
        slect_all_feature = [upsample(f, (64, 64)) for f in slect_all_feature]
        slect_all_feature = torch.cat(slect_all_feature, dim=1)
        slect_all_feature = self.select_all_feature_proj(slect_all_feature)

        high_feature = torch.cat((res_feature, slect_all_feature), dim=1)
        high_feature = self.feature_merged_proj(high_feature).flatten(2).permute(0,2,1)
        print("high_feature-------------", high_feature.shape)

        # for j in select_original:
        #     print("j",j.shape)
        select_d_proj = [self.select_d_proj[i](j) for i, j in enumerate(select_original)]
        encode_result = self.encoder(high_feature, x_new, select_d_proj)        
        
        # encode_result = self.encoder(x_new, new_select_d_features)
    
        # unet structure
        result = None
        resultd = None
        if self.training:
            x_downsample = []
            for i in range(len(res)):
                # order0, ---shapetorch.Size([10, 256, 64, 64])
                # order1, ---shapetorch.Size([10, 512, 32, 32])
                # order2, ---shapetorch.Size([10, 1024, 16, 16])
                # order3, ---shapetorch.Size([10, 2048, 8, 8])
                # print("order{}, ---shape{}".format(i,res[i].shape))
                x = self.layers_convbn[i](res[i])
                x = x.flatten(2).permute(0, 2, 1)
                x_downsample.append(x)

            y_downsample_d = []
            for i in range(len(resd)):
                y = self.layers_convbn_d[i](resd[i])
                y = y.flatten(2).permute(0, 2, 1)
                y_downsample_d.append(y) 
            result, resultd = self.decoder(x, x_downsample, y, y_downsample_d)

        # encode_result_downsample = encode_result[::-1]
        # for i in encode_result_downsample:
        #     print("iiiiiiiiiiii",i.shape)
        
        # for i in encode_result:
        #     print("iiiiiiiiiiii----",i.shape)        

        x, feature = self.final_decoder(encode_result[-1], encode_result)
        return result, resultd, x, basefeature, detailfeature, basefeature_d, detailfeature_d, feature
        # return result, resultd, x, basefeature, detailfeature, basefeature_d, detailfeature_d, x_global_features_merged, y_global_features_merged,img_querys[1],img_querys[2],img_querys[3],y_local_features_embed[1],y_local_features_embed[2],y_local_features_embed[3], slect_basefeature_d.flatten(2).permute(0,2,1), basefeature.flatten(2).permute(0,2,1),slect_detailfeature_d.flatten(2).permute(0,2,1), detailfeature.flatten(2).permute(0,2,1), detailfeature_d.flatten(2).permute(0,2,1),select_original,slect_global_feature.flatten(2).permute(0,2,1)
        # return result, resultd, x, basefeature, detailfeature, basefeature_d, detailfeature_d, x_global_features_merged, y_global_features_merged,slect_detailfeature_d,detailfeature.flatten(2).permute(0,2,1)
# swsl_resnet18
if  __name__ == "__main__":
    from tqdm import tqdm
    tensor_random = torch.rand([8, 3, 256, 256])
    tensor_random_d = torch.rand([8, 1, 256, 256])
    tensor_random = tensor_random.cuda()
    tensor_random_d = tensor_random_d.cuda()
    model = UNetFormer()
    model = model.cuda()
    model.train()
    # model.eval()

    x, y = model(tensor_random, tensor_random_d)

# class ConvBN(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
#         super(ConvBN, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=1),
#             norm_layer(out_channels)
#         )

# res1 torch.Size([8, 256, 64, 64])
# res2 torch.Size([8, 512, 32, 32])
# res3 torch.Size([8, 1024, 16, 16])
# res4 torch.Size([8, 2048, 8, 8])

# 256
# res1 torch.Size([8, 256, 128, 128])
# res2 torch.Size([8, 512, 64, 64])
# res3 torch.Size([8, 1024, 32, 32])
# res4 torch.Size([8, 2048, 16, 16])

# 224
# res1 torch.Size([8, 256, 56, 56])
# res2 torch.Size([8, 512, 28, 28])
# res3 torch.Size([8, 1024, 14, 14])
# res4 torch.Size([8, 2048, 7, 7])

