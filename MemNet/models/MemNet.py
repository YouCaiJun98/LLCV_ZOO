import torch
import torch.nn as nn

__all__ = ['MemNet', 'memnet']


# ---------------- Basic Building Blocks ---------------- #
def default_conv(in_chs, out_chs, kernel_size=3, bias=True):
    return nn.Conv2d(in_chs, out_chs, kernel_size,
                     padding=(kernel_size//2), bias=bias)

class BNReLUConv(nn.Sequential):
    def __init__(self, conv, in_feat, out_feat, kernel_size=3, bias=True, BN=True, act=nn.ReLU(True)):
        super(BNReLUConv, self).__init__()
        if BN:
            self.add_module('BN', nn.BatchNorm2d(in_feat))
        if act:
            self.add_module('Act', act)
        self.add_module('Conv', conv(in_feat, out_feat, kernel_size, bias))


class ResidualBlock(nn.Module):
    def __init__(self, conv, in_feat, out_feat, kernel_size=3, bias=True, BN=True, act=nn.ReLU(True)):
        super(ResidualBlock, self).__init__()
        self.bnreluconv_1 = BNReLUConv(conv, in_feat, out_feat, kernel_size, bias, BN, act)
        self.bnreluconv_2 = BNReLUConv(conv, in_feat, out_feat, kernel_size, bias, BN, act)

    def forward(self, x):
        res = x
        out = self.bnreluconv_1(x)
        out = self.bnreluconv_2(out)
        out = out + res
        return out


class MemoryBlock(nn.Module):
    # num_prememblock denotes the number of long term memory channels before this block.
    def __init__(self, channels, num_resblock, num_prememblock, BN):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(default_conv, channels, channels, kernel_size=3, BN=BN) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv(default_conv, (num_resblock + num_prememblock) * channels, channels, kernel_size=1, BN=BN)

    def forward(self, x, long_mem):
        short_mem = []
        for layer in self.recursive_unit:
            x = layer(x)
            short_mem.append(x)

        gate_out = self.gate_unit(torch.cat(long_mem + short_mem, 1))

        return gate_out



# ---------------- Model Construction ---------------- #
class MemNet(nn.Module):
    def __init__(self, model_settings: dict):
        super(MemNet, self).__init__()
        self.BN               = model_settings['BN']
        self.bias             = model_settings['bias']
        self.act_func         = model_settings['act_func']
        self.in_channels      = model_settings['in_channels']
        self.out_channels     = model_settings['out_channels']
        self.init_channels    = model_settings['init_channels']
        self.num_resblocks    = model_settings['num_resblocks']
        self.num_memblocks    = model_settings['num_memblocks']
        self.multi_supervised = model_settings['multi_supervised']

        if self.act_func == 'relu':
            self.act = nn.ReLU(inplace=True)
        self.conv = default_conv

        self.fenet = nn.Conv2d(self.in_channels, self.init_channels, kernel_size=3, stride=1, padding=1)
        self.mem_blocks = nn.ModuleList(
            # it should be noticed that the extracted features are also taken as long-term memory
            [MemoryBlock(self.init_channels, self.num_resblocks, i+1, self.BN) for i in range(self.num_memblocks)]
        )
        self.reconnet = nn.Conv2d(self.init_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if self.multi_supervised:
            self.factors = nn.Parameter(torch.ones(self.num_memblocks))

    def forward(self, x):
        res = x
        out = self.fenet(x)
        long_mem = [out]
        for memory_block in self.mem_blocks:
            out = memory_block(out, long_mem)
            long_mem.append(out)
        if not self.multi_supervised:
            out = self.reconnet(out)
            out = res + out
            return out
        else:
            # supervised multi-branch features
            outs = []
            for i in range(1, self.num_memblocks+1):
                # different reconstruction branches
                outs.append(x + self.reconnet(long_mem[i]))
            final = 0
            for i in range(self.num_memblocks):
                final += self.factors[i] * outs[i]
            return final, outs



# ---------------- Model Instantiation ---------------- #
def memnet(cfgs, ckpt_path: str=None, device=None):
    model = MemNet(cfgs)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f'load pretrained checkpoint from: {ckpt_path}')
        model.load_state_dict(ckpt['state_dict'])
    return model
