import torch
import torch.nn as nn

__all__ = ['DnCNN', 'dncnn']


# ---------------- Basic Building Blocks ---------------- #
def default_conv(in_chs, out_chs, kernel_size, bias=True):
    return nn.Conv2d(in_chs, out_chs, kernel_size,
                     padding=(kernel_size//2), bias=bias)


def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.

    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ConvBNReLU(nn.Module):
    def __init__(self, conv, in_feat, out_feat, kernel_size=3, bias=True, BN=False, act=nn.ReLU(True)):
        super(ConvBNReLU, self).__init__()
        m = []
        m.append(conv(in_feat, out_feat, kernel_size, bias=bias))
        if BN:
            m.append(nn.BatchNorm2d(out_feat))
        if act:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)

# ---------------- Model Construction ---------------- #
class DnCNN(nn.Module):
    def __init__(self, model_settings:dict):
        super(DnCNN, self).__init__()
        self.BN            = model_settings['BN']
        self.bias          = model_settings['bias']
        self.act_func      = model_settings['act_func']
        self.conv_layers   = model_settings['conv_layers']
        self.in_channels   = model_settings['in_channels']
        self.out_channels  = model_settings['out_channels']
        self.init_channels = model_settings['init_channels']

        if self.act_func == 'relu':
            self.act = nn.ReLU(inplace=True)
        self.conv = default_conv

        m_head = ConvBNReLU(self.conv, self.in_channels, self.init_channels, bias=self.bias, BN=False, act=self.act)
        m_body = [ConvBNReLU(self.conv, self.init_channels, self.init_channels, bias=self.bias, BN=self.BN, act=self.act) for _ in range(self.conv_layers - 2)]
        m_tail = ConvBNReLU(self.conv, self.init_channels, self.out_channels, bias=self.bias, BN=False, act=False)
        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        res = self.model(x)
        return x - res

# ---------------- Model Instantiation ---------------- #
def dncnn(cfgs, ckpt_path: str=None, device=None):
    model = DnCNN(cfgs)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f'load pretrained checkpoint from: {ckpt_path}')
        model.load_state_dict(ckpt['state_dict'])
    return model 
