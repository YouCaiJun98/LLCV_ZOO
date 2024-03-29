import math

import torch
import torch.nn as nn

from .arch_utils import *

__all__ = ['nafnet', 'baselinenet', 'plainnet']

# TODO: 1. update different types of scaling methods


# --------------------------- Function Blocks --------------------------- #
class ChannelAttention(nn.Module):
    def __init__(self, ch):
        super(ChannelAttention, self).__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True), # here we adopt leakyrelu instead of relu
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.body(x) * x


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, ch):
        super(SimplifiedChannelAttention, self).__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.body(x) * x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# --------------------------- Building Blocks --------------------------- #
class PlainNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=2, kernel_size=3, stride=1, rc='skip'):
        # rc denotes the residual connection type.
        super(PlainNetBlock, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.dw_ch  = expand_ratio * in_ch
        self.rc     = rc
        assert rc in ['skip', 'conv']

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_ch, self.dw_ch, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.dw_ch, self.dw_ch, kernel_size=kernel_size, stride=stride, padding=1, groups=self.dw_ch),
            nn.LeakyReLU(),
            nn.Conv2d(self.dw_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        if rc == 'skip' and in_ch == out_ch:
            self.rc_1 = nn.Sequential()
            self.rc_2 = nn.Sequential()
        else:
            self.rc_1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
            self.rc_2 = nn.Conv2d(out_ch,out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.block_1(x) + self.rc_1(x)
        out2 = self.block_2(out1) + self.rc_2(out1)

        return out2


class BaselineBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=2, kernel_size=3, stride=1, rc='skip'):
        # rc denotes the residual connection type.
        super(BaselineBlock, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.dw_ch  = expand_ratio * in_ch
        self.rc     = rc
        assert rc in ['skip', 'conv']

        self.block_1 = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, self.dw_ch, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.dw_ch, self.dw_ch, kernel_size=kernel_size, stride=stride, padding=1, groups=self.dw_ch),
            nn.GELU(),
            ChannelAttention(self.dw_ch),
            nn.Conv2d(self.dw_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        self.block_2 = nn.Sequential(
            LayerNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        if rc == 'skip' and in_ch == out_ch:
            self.rc_1 = nn.Sequential()
            self.rc_2 = nn.Sequential()
        else:
            self.rc_1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
            self.rc_2 = nn.Conv2d(out_ch,out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.block_1(x) + self.rc_1(x)
        out2 = self.block_2(out1) + self.rc_2(out1)

        return out2


class NAFNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=2, FFN_ratio=2, kernel_size=3, stride=1, rc='skip', drop_out_ratio=0.):
        super(NAFNetBlock, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.dw_ch  = expand_ratio * in_ch
        self.ffn_ch = FFN_ratio * in_ch
        self.rc     = rc
        assert rc in ['skip', 'conv']

        self.block_1 = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, self.dw_ch, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.dw_ch, self.dw_ch, kernel_size=kernel_size, stride=stride, padding=1, groups=self.dw_ch),
            SimpleGate(),
            SimplifiedChannelAttention(self.dw_ch // 2), # because simplegate will reduce ch by 2x
            nn.Conv2d(self.dw_ch // 2, in_ch, kernel_size=1, stride=1, padding=0)
        )
        self.block_2 = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, self.ffn_ch, kernel_size=1, stride=1, padding=0),
            SimpleGate(),
            nn.Conv2d(self.ffn_ch // 2, out_ch, kernel_size=1, stride=1, padding=0)
        )
        self.dropout1 = nn.Dropout(drop_out_ratio) if drop_out_ratio > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_ratio) if drop_out_ratio > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, in_ch, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, out_ch, 1, 1)), requires_grad=True)

        if rc == 'skip' and in_ch == out_ch:
            self.rc_1 = nn.Sequential()
            self.rc_2 = nn.Sequential()
        else:
            self.rc_1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
            self.rc_2 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        res1 = self.rc_1(x)
        int1 = self.dropout1(self.block_1(x)) * self.beta
        out1 = int1 + res1
        res2 = self.rc_2(out1)
        int2 = self.dropout2(self.block_2(out1)) * self.gamma
        out2 = int2 + res2

        return out2



# --------------------------- Model Definition --------------------------- #
class NAFNet(nn.Module):
    def __init__(self, model_cfg, BlockType=NAFNetBlock):
        super(NAFNet, self).__init__()
        self.in_ch = model_cfg['in_channels']
        self.out_ch = model_cfg['out_channels']
        self.width = model_cfg['width']
        self.encoder_block = model_cfg['encoder_block']
        self.decoder_block = model_cfg['decoder_block']
        self.middle_block  = model_cfg['middle_block']
        self.res_connect  = model_cfg['res_connect']
        self.long_connect = model_cfg['long_connect']
        self.Head = model_cfg['head']
        assert self.res_connect in ['conv', 'skip']
        assert self.long_connect in ['cat', 'add']

        if self.Head:
            self.head = nn.Conv2d(self.in_ch, self.width, kernel_size=3, stride=1, padding=1)

        self.encoder    = nn.ModuleList()
        self.decoder    = nn.ModuleList()
        self.upscales   = nn.ModuleList()
        self.downscales = nn.ModuleList()

        width = self.width
        for num in self.encoder_block:
            self.encoder.append(
                nn.Sequential(
                    *[BlockType(width, width, rc=self.res_connect) for _ in range(num)]
                )
            )
            self.downscales.append(
                nn.Conv2d(width, width*2, kernel_size=2, stride=2)
            )
            width *= 2

        self.middle = nn.Sequential(
            *[BlockType(width, width, rc=self.res_connect) for _ in range(self.middle_block)]
        )

        for num in self.decoder_block:
            self.upscales.append(
                nn.Sequential(
                    nn.Conv2d(width, width*2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            width = width // 2
            self.decoder.append(
                nn.Sequential(
                    *([BlockType(width + (self.long_connect == 'cat') * width, width, rc=self.res_connect)] + \
                      [BlockType(width, width, rc=self.res_connect) for _ in range(num-1)])
                )
            )


        self.tail = nn.Conv2d(self.width, self.out_ch, kernel_size=3, stride=1, padding=1)

        self.padder_size = 2 ** len(self.encoder_block)

    def forward(self, x):
        n, c, h, w = x.size()
        inp = self.check_img_size(x)

        if self.Head:
            x = self.head(inp)
        else:
            x = inp

        encs = []
        for encoder, downscale in zip(self.encoder, self.downscales):
            x = encoder(x)
            encs.append(x)
            x = downscale(x)

        x = self.middle(x)

        for decoder, upscale, enc_skip in zip(self.decoder, self.upscales, encs[::-1]):
            x = upscale(x)
            if self.long_connect == 'add':
                x = x + enc_skip
            else:
                x = torch.cat([x, enc_skip], 1)
            x = decoder(x)

        x = self.tail(x)
        x = x + inp

        return x[:,:,:h,:w]


    def check_img_size(self, x):
        n, c, h, w = x.size()
        h_pad = (self.padder_size - h % self.padder_size) % self.padder_size
        w_pad = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, w_pad, 0, h_pad))
        return x



# --------------------------- Model Instantiation --------------------------- #
def plainnet(model_cfg, model_path: str=None, device=None):
    model = NAFNet(model_cfg, BlockType=PlainNetBlock)
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        print(f'load pretrained checkpoint from: {model_path}')
        model.load_state_dict(state_dict)
    return model


def baselinenet(model_cfg, model_path: str=None, device=None):
    model = NAFNet(model_cfg, BlockType=BaselineBlock)
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        print(f'load pretrained checkpoint from: {model_path}')
        model.load_state_dict(state_dict)
    return model


def nafnet(model_cfg, model_path: str=None, device=None):
    model = NAFNet(model_cfg)
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        print(f'load pretrained checkpoint from: {model_path}')
        model.load_state_dict(state_dict)
    return model

