import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

__all__ = ['SGN', 'sgn_l1', 'sgn_l2', 'sgn_l3', 'sgn_l4']

# ---------------- Basic Operation ---------------- #
class Concate(nn.Module):
    def __init__(self):
        super(Concate, self).__init__()

    def forward(self, x, y):
        return torch.cat((x, y), 1)

def default_conv(in_chs, out_chs, kernel_size, bias=True):
    return nn.Conv2d(in_chs, out_chs, kernel_size,
                     padding=(kernel_size//2), bias=bias)

class ConvBNReLU(nn.Module):
    def __init__(self, conv, in_feat, out_feat, kernel_size, bias=True, BN=False, act=nn.ReLU(True)):
        super(ConvBNReLU, self).__init__()
        m = []
        m.append(conv(in_feat,out_feat,kernel_size,bias=bias))
        if BN:
            m.append(nn.BatchNorm2d(out_feat))
        if act:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)

class ResidualBlock(nn.Module):
    def __init__(self, g, conv, in_feat, out_feat, kernel_size, bias=True, BN=False, act=nn.ReLU(True),
                 res=True):
        super(ResidualBlock, self).__init__()
        self.res = res
        # In fact in_channels seems to always be the same as out_feat.
        m = [ConvBNReLU(conv, in_feat, out_feat, kernel_size, bias, BN, act) for _ in range(g)]
        # the last layer in the residual block has no activation functions.
        m.append(conv(in_feat, out_feat, kernel_size, bias))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = x
        if self.res:
            return self.body(x) + res
        return self.body(x)


# ---------------- Main Model ---------------- #

class SGN(nn.Module):
    def __init__(self, hyper_params, conv=default_conv):
        super(SGN, self).__init__()
        # hyper-parameters
        self.g = hyper_params['g']
        self.m = hyper_params['m']
        self.level = hyper_params['level']
        self.kernel_size = hyper_params['kernel_size']
        # dedicated for raw image processing usage.
        self.in_channels =  hyper_params['in_channels']
        self.out_channels = hyper_params['out_channels']
        # initial channels
        self.init_channels = hyper_params['init_channels']

        assert self.level in [0, 1, 2, 3, 4], "The specified level {} is not supported!".format(self.level)

        # op settings
        if hyper_params['act'] == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=False)
        elif hyper_params['act'] == 'relu':
            self.act = nn.ReLU(True)

        self.BN = hyper_params['BN']

        self.fusion = Concate()

        self.upsample = nn.PixelShuffle(2)
        self.downsample = nn.PixelUnshuffle(2)


        # basic block definition
        self.l3_head = ConvBNReLU(conv, self.in_channels * 64, self.init_channels * 8, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)
        self.l3_body = ResidualBlock(self.g, conv, self.init_channels * 8, self.init_channels * 8, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)
        self.l3_tail = ConvBNReLU(conv, self.init_channels * 8, self.init_channels * 8, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)

        self.l2_head = ConvBNReLU(conv, self.in_channels * 16, self.init_channels * 4, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)
        self.l2_merge= ConvBNReLU(conv, self.init_channels * 6, self.init_channels * 4, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)
        self.l2_body = ResidualBlock(self.g, conv, self.init_channels * 4, self.init_channels * 4, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)
        self.l2_tail = ConvBNReLU(conv, self.init_channels * 4, self.init_channels * 4, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)

        self.l1_head = ConvBNReLU(conv, self.in_channels * 4, self.init_channels * 2, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)
        self.l1_merge= ConvBNReLU(conv, self.init_channels * 3, self.init_channels * 2, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)
        self.l1_body = ResidualBlock(self.g, conv, self.init_channels * 2, self.init_channels * 2, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)
        self.l1_tail = ConvBNReLU(conv, self.init_channels * 2, self.init_channels * 2, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)

        self.l0_head = ConvBNReLU(conv, self.in_channels, self.init_channels, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)
        self.l0_merge= ConvBNReLU(conv, int(self.init_channels * 1.5), self.init_channels, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act)
        self.l0_body = ResidualBlock(self.m, conv, self.init_channels, self.init_channels, self.kernel_size,
                                  bias=True, BN=self.BN, act=self.act, res=False)
        self.l0_tail = ConvBNReLU(conv, self.init_channels, self.out_channels, self.kernel_size,
                                  bias=True, BN=False, act=None)

        self.init_params()


    def init_params(self):
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                self.weight_init(m)

    @staticmethod
    def weight_init(m):
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)

    def forward(self, x):
        # though we may not need that many levels.
        l1_init = self.downsample(x)
        l2_init = self.downsample(l1_init)
        l3_init = self.downsample(l2_init)
        l4_init = self.downsample(l3_init)

        l3 = self.l3_head(l3_init)
        l3 = self.l3_body(l3)
        l3 = self.l3_tail(l3)
        l3 = self.upsample(l3)

        l2 = self.l2_head(l2_init)
        l2 = self.l2_merge(self.fusion(l2, l3))
        l2 = self.l2_body(l2)
        l2 = self.l2_tail(l2)
        l2 = self.upsample(l2)

        l1 = self.l1_head(l1_init)
        l1 = self.l1_merge(self.fusion(l1, l2))
        l1 = self.l1_body(l1)
        l1 = self.l1_tail(l1)
        l1 = self.upsample(l1)

        l0 = self.l0_head(x)
        l0 = self.l0_merge(self.fusion(l0, l1))
        l0 = self.l0_body(l0)
        l0 = self.l0_tail(l0)

        out = l0 + x

        return out

# ---------------- Model Instantiation ---------------- #

def sgn_l1():
    return

def sgn_l2():
    return

def sgn_l3(cfgs):
    return SGN(cfgs)

def sgn_l4():
    return
