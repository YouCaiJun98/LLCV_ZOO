import torch
import torch.nn as nn
import math

__all__ = ['LSID', 'unet_mosp', 'module_name']

module_name = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
               'conv4_1', 'conv4_2', 'conv5_1', 'conv5_2', 'up6', 'conv6_1',
               'conv6_2', 'up7', 'conv7_1', 'conv7_2', 'up8', 'conv8_1', 'conv8_2',
               'up9', 'conv9_1', 'conv9_2']

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


class LSID(nn.Module):
    def __init__(self, model_cfg):
        super(LSID, self).__init__()
        self.block_size = model_cfg['block_size']
        self.in_channels = model_cfg['in_channels']
        self.out_channels = model_cfg['out_channels']
        self.init_channels = model_cfg['width']
        assert type(self.init_channels) in [int, float], 'Number of Base Channel should be an integer or a float.'

        self.conv1_1 = nn.Conv2d(self.in_channels, 22, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(22, 24, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(24, 47, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(47, 41, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(41, 95, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(95, 94, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(94, 187, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(187, 190, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5_1 = nn.Conv2d(190,  390, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(390, 376, kernel_size=3, stride=1, padding=1, bias=True)

        self.up6 = nn.ConvTranspose2d(376, 156, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(346, 196, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(196, 204, kernel_size=3, stride=1, padding=1, bias=True)

        self.up7 = nn.ConvTranspose2d(204, 81, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(175, 93, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(93, 98, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(98, 40, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(81, 48, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(48, 42, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(42, 24, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(24, 23, kernel_size=3, stride=1, padding=1, bias=True)

        # The old implementation seems to set out channels with the size of the Bayer Pattern
        out_channel = self.out_channels if self.out_channels else 3 * (self.block_size ** 2)
        self.conv10 = nn.Conv2d(23, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def _ch(self, width):
        ch = width * self.init_channels
        return ch if type(self.init_channels) is int else math.ceil(ch)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)


        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)

        x = self.up6(x)
        x = torch.cat((x, conv4), 1)
        # x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)

        x = self.up7(x)
        x = torch.cat((x, conv3), 1)
        # x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)

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

def unet_mosp(model_cfg, model_path: str=None, device=None):
    model = LSID(model_cfg)
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        print(f'load pretrained checkpoint from: {model_path}')
        model.load_state_dict(state_dict)
    return model

