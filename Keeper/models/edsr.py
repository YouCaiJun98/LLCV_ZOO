import math

import torch
import torch.nn as nn

__all__ = ['EDSR', 'edsr']



# --------------------------- Function Blocks --------------------------- #
def default_conv(in_chs, out_chs, kernel_size, bias=True):
    return nn.Conv2d(
        in_chs, out_chs, kernel_size, padding=(kernel_size//2), bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=[0.4488, 0.4371, 0.4040],
                 rgb_std=[1.0, 1.0, 1.0], sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class EDSR_block(nn.Module):
    def __init__(self, conv, width, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True), res_scale=1.):
        super(EDSR_block, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(width, width, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(width))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, width, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(width, 4 * width, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    # TODO: why BN here?
                    m.append(nn.BatchNorm2d(width))
                if act == 'relu':
                    m.append(nn.ReLU(True))

        elif scale == 3:
            m.append(conv(width, 9 * width, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(width))
            if act == 'relu':
                m.append(nn.ReLU(True))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



# --------------------------- Model Body --------------------------- #
class EDSR(nn.Module):
    def __init__(self, model_cfg):
        super(EDSR, self).__init__()
        self.scale = model_cfg.get('scale', 2)
        self.width = model_cfg.get('width', 64)
        self.blocks = model_cfg.get('blocks', 16)
        self.in_chs = model_cfg.get('in_channels', 3)
        self.out_chs = model_cfg.get('out_channels', 3)
        self.act = model_cfg.get('activation', 'ReLU')
        self.res_scale = model_cfg.get('res_scale', 1.)
        self.rgb_range = model_cfg.get('rgb_range', 255.)
        self.rgb_mean = model_cfg.get('rgb_mean', [0.4488, 0.4371, 0.4040])
        conv = default_conv
        kernel_size = 3
        act = None
        if self.act == 'ReLU':
            act = nn.ReLU(True)
        else:
            raise NotImplementedError(f'Activation Type {self.act} is not supported yet.')

        self.sub_mean = MeanShift(self.rgb_range, self.rgb_mean)
        self.add_mean = MeanShift(self.rgb_range, self.rgb_mean, sign=1)


        m_head = [conv(self.in_chs, self.width, kernel_size)]
        m_body = [
            EDSR_block(
                conv, self.width, kernel_size, act=act, res_scale=self.res_scale
            ) for _ in range(self.blocks)
        ]
        m_body.append(conv(self.width, self.width, kernel_size))
        m_tail = [
            Upsampler(conv, self.scale, self.width, act=False),
            conv(self.width, self.out_chs, kernel_size)
        ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):
        # x *= 255.
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        # x /= 255.
        return x


    # For x3/x4 pretrain purpose.
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))



# --------------------------- Instantiation --------------------------- #
def edsr(model_cfg, model_path: str=None, device=None):
    model = EDSR(model_cfg)
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        print(f'load pretrained checkpoint from: {model_path}')
        model.load_state_dict(state_dict)
    return model
