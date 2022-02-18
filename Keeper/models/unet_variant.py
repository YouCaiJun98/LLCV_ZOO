import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import utils

__all__ = ['LSID', 'lsid']

def pixel_shuffle(input, upscale_factor, depth_first=False):
    r"""Rearranges elements in a tensor of shape :math:`[*, C*r^2, H, W]` to a
    tensor of shape :math:`[C, H*r, W*r]`.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Tensor): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples::

        >>> ps = nn.PixelShuffle(3)
        >>> input = torch.empty(1, 9, 4, 4)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])
    """
    batch_size, channels, in_height, in_width = input.size()
    channels //= upscale_factor ** 2

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    if not depth_first:
        input_view = input.contiguous().view(
            batch_size, channels, upscale_factor, upscale_factor,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        return shuffle_out.view(batch_size, channels, out_height, out_width)
    else:
        input_view = input.contiguous().view(batch_size, upscale_factor, upscale_factor, channels, in_height, in_width)
        shuffle_out = input_view.permute(0, 4, 1, 5, 2, 3).contiguous().view(batch_size, out_height, out_width,
                                                                             channels)
        return shuffle_out.permute(0, 3, 1, 2)

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_r=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

# outchannels are fixed according to the given ratio.
class GhostModule(nn.Module):
    def __init__(self, in_chs, ratio=4, pconv_kernel=1, gconv_kernel=3, stride=1, act=nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        super(GhostModule, self).__init__()
        assert in_chs % ratio == 0, 'The input channls is not divisible by the given ratio.'
        med_chs = in_chs // ratio
        new_chs = med_chs * (ratio - 1)
        self.primary_convs = nn.Sequential(
            nn.Conv2d(in_chs, med_chs, pconv_kernel, stride, (pconv_kernel - 1) //2, bias=False),
            act if act is not None else nn.Sequential(),
        )
        self.ghost_convs = nn.Sequential(
            nn.Conv2d(med_chs, new_chs, gconv_kernel, 1, (gconv_kernel - 1)//2, groups=med_chs, bias=False),
            act if act is not None else nn.Sequential(),
        )

    def forward(self, x):
        x1  = self.primary_convs(x)
        x2  = self.ghost_convs(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

class LSID(nn.Module):
    def __init__(self, inchannel=4, block_size=2, outchannel=None, ratio=32, gratio=4):
        super(LSID, self).__init__()
        self.block_size = block_size
        self.outchannel = outchannel
        assert type(ratio) is int, 'Channel multiplier should be a integer.'

        self.conv1_1 = nn.Conv2d(inchannel, 1*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(1*ratio, 1*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.skip_1 = GhostModule(1*ratio, gratio)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(1*ratio, 2*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(2*ratio, 2*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.skip_2 = GhostModule(2*ratio, gratio)

        self.conv3_1 = nn.Conv2d(2*ratio, 4*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(4*ratio, 4*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.skip_3 = GhostModule(4*ratio, gratio)

        self.conv4_1 = nn.Conv2d(4*ratio, 8*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(8*ratio, 8*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.skip_4 = GhostModule(8*ratio, gratio)

        self.conv5_1 = nn.Conv2d(8*ratio, 16*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(16*ratio, 16*ratio, kernel_size=3, stride=1, padding=1, bias=True)

        self.up6 = nn.ConvTranspose2d(16*ratio, 8*ratio, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(16*ratio, 8*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(8*ratio, 8*ratio, kernel_size=3, stride=1, padding=1, bias=True)

        self.up7 = nn.ConvTranspose2d(8*ratio, 4*ratio, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(8*ratio, 4*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(4*ratio, 4*ratio, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(4*ratio, 2*ratio, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(4*ratio, 2*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(2*ratio, 2*ratio, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(2*ratio, 1*ratio, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(2*ratio, 1*ratio, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(1*ratio, 1*ratio, kernel_size=3, stride=1, padding=1, bias=True)

        # The old implementation seems to set out channels with the size of the Bayer Pattern
        out_channel = 3 * self.block_size * self.block_size if not self.outchannel else self.outchannel
        self.conv10 = nn.Conv2d(1*ratio, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #import ipdb; ipdb.set_trace()
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) 
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        import ipdb
        # ipdb.set_trace()
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = self.skip_1(x) 
        x = self.maxpool(x)


        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = self.skip_2(x)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = self.skip_3(x)
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = self.skip_4(x)
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)

        x = self.up6(x)
        # ipdb.set_trace()
        x = torch.cat((x, conv4), 1)
        # x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)

        x = self.up7(x)
        x = torch.cat((x, conv3), 1)
        #x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)
        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        x = torch.cat((x, conv2), 1)
        # x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)
        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        x = torch.cat((x, conv1), 1)
        # x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)
        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)

        #depth_to_space_conv = pixel_shuffle(x, upscale_factor=self.block_size, depth_first=True)

        return x # depth_to_space_conv

def lsid(pretrained:bool=False, model_path:str=None, device=None,**kwargs):
    model = LSID(**kwargs)
    if pretrained and model_path:
        state_dict = torch.load(model_path, map_location=device)
        print(f'load pretrained checkpoint from: {model_path}')
        model.load_state_dict(state_dict)
    return model

