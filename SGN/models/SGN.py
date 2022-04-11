import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

__all__ = ['SGN', 'sgn']

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

class SGNLevel(nn.Module):
    def __init__(self, level: int, level_type: str, body_length, conv, in_ch, out_ch=None, init_ch=32,
                 kernel_size=3, bias=True, BN=False, act=nn.ReLU(True), info_merge=Concate()):
        super(SGNLevel, self).__init__()
        assert level_type in ['top', 'middle', 'bottom'], "We only categorize one level into top/middle/bottom type!"
        assert type(level) == int
        self.level = level
        self.body_length = body_length

        if level_type == 'bottom' or (level_type == 'top' and level == 0):
            tail_act = None
        else:
            tail_act = act

        self.fusion = info_merge
        self.head = ConvBNReLU(conv, in_ch * (4**level), init_ch * (2**level), kernel_size,
                                  bias=bias, BN=BN, act=act)
        self.body = ResidualBlock(body_length, conv, init_ch * (2**level), init_ch * (2**level), kernel_size,
                                  bias=True, BN=BN, act=act, res=(level_type != 'bottom'))
        self.tail = ConvBNReLU(conv, init_ch * (2**level), out_ch if out_ch else init_ch * (2**level), kernel_size,
                                  bias=True, BN=BN, act=tail_act)
        if level_type != 'top':
            self.neck = ConvBNReLU(conv, int(init_ch * (2**level) * 1.5), init_ch * (2**level), kernel_size,
                                  bias=True, BN=BN, act=act)

    def forward(self, x, guiding_info):
        out = self.head(x)
        if hasattr(self, 'neck'):
            out = self.neck(self.fusion(out, guiding_info))
        out = self.body(out)
        out = self.tail(out)
        return out

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
        if self.level == 4:
            self.level_4 = SGNLevel(level=4, level_type='top', body_length=self.g, conv=conv, in_ch=self.in_channels,
                                    init_ch=self.init_channels, kernel_size=self.kernel_size, BN=self.BN, act=self.act,
                                    info_merge=self.fusion)
        if self.level >= 3:
            self.level_3 = SGNLevel(level=3, level_type='top' if self.level == 3 else 'middle', body_length=self.g,
                                    conv=conv, in_ch=self.in_channels, init_ch=self.init_channels,
                                    kernel_size=self.kernel_size, BN=self.BN, act=self.act, info_merge=self.fusion)

        if self.level >= 2:
            self.level_2 = SGNLevel(level=2, level_type='top' if self.level == 2 else 'middle', body_length=self.g,
                                    conv=conv, in_ch=self.in_channels, init_ch=self.init_channels,
                                    kernel_size=self.kernel_size, BN=self.BN, act=self.act, info_merge=self.fusion)

        if self.level >= 1:
            self.level_1 = SGNLevel(level=1, level_type='top' if self.level == 1 else 'middle', body_length=self.g,
                                    conv=conv, in_ch=self.in_channels, init_ch=self.init_channels,
                                    kernel_size=self.kernel_size, BN=self.BN, act=self.act, info_merge=self.fusion)

        self.level_0 = SGNLevel(level=0, level_type='top' if self.level == 0 else 'bottom', body_length=self.m,
                                    conv=conv, in_ch=self.in_channels, out_ch=self.out_channels, init_ch=self.init_channels,
                                    kernel_size=self.kernel_size, BN=self.BN, act=self.act, info_merge=self.fusion)

        '''
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
        '''
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

        if self.level == 4:
            l4 = self.level_4(l4_init, None)
            l4 = self.upsample(l4)
        if self.level >= 3:
            l3 = self.level_3(l3_init, None if self.level == 3 else l4)
            l3 = self.upsample(l3)
        if self.level >= 2:
            l2 = self.level_2(l2_init, None if self.level == 2 else l3)
            l2 = self.upsample(l2)
        if self.level >= 1:
            l1 = self.level_1(l1_init, None if self.level == 1 else l2)
            l1 = self.upsample(l1)
        l0 = self.level_0(x, None if self.level == 0 else l1)
        out = l0 + x
        return out

# ---------------- Model Instantiation ---------------- #
def sgn(cfgs, ckpt_path: str=None, device=None):
    model = SGN(cfgs)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f'load pretrained checkpoint from: {ckpt_path}')
        model.load_state_dict(ckpt['state_dict'])
    return SGN(cfgs)
