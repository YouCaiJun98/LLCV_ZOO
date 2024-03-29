import torch
import torch.nn as nn
import math

__all__ = ['resunet', 'dconvunet', 'module_name']

module_name = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
               'conv4_1', 'conv4_2', 'conv5_1', 'conv5_2', 'up6', 'conv6_1',
               'conv6_2', 'up7', 'conv7_1', 'conv7_2', 'up8', 'conv8_1', 'conv8_2',
               'up9', 'conv9_1', 'conv9_2']


# --------------------------- Building Blocks --------------------------- #
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

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, res_connect='conv'):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        assert res_connect in ['skip', 'conv']

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.LeakyReLU(inplace=True),
        )
        if stride == 1 and res_connect == 'skip':
            self.res_connect = nn.Sequential()
        else:
            self.res_connect = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        out1 = self.res_connect(x)
        out2 = self.block(x)
        return out1 + out2

class DConvBlock(nn.Module):
    # a.k.a. Building Block of PlainNet in NAFNet, the only difference lies in here we adopt LeakyReLU.
    # wait a minute! the channels might be different in the second block.
    def __init__(self, in_ch, out_ch, DW_Expand=2, kernel_size=3, stride=1, res_connect='skip'):
        super(DConvBlock, self).__init__()
        self.stride = stride
        self.in_ch  = in_ch
        self.dw_ch  = in_ch * DW_Expand
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        assert res_connect in ['skip', 'conv']

        if stride == 1 and res_connect == 'skip':
            self.res_connect_1 = nn.Sequential()
            self.res_connect_2 = nn.Sequential()
        else:
            # TODO: check the proper channel size.
            self.res_connect_1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)
            self.res_connect_2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=stride, padding=0)

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_ch, self.dw_ch, kernel_size=1, stride=1, padding=0),
            # depthwise conv
            nn.Conv2d(self.dw_ch, self.dw_ch, kernel_size=kernel_size, stride=stride, padding=1, groups=self.dw_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.dw_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )


    def forward(self, x):
        out1 = self.block_1(x) + self.res_connect_1(x)
        out  = self.block_2(out1) + self.res_connect_2(out1)
        return out


# --------------------------- Model Definition --------------------------- #

class BlockUNet(nn.Module):
    def __init__(self, model_cfg, BlockType=ResBlock):
        super(BlockUNet, self).__init__()
        self.block_size = model_cfg['block_size']
        self.res_connect = model_cfg['res_connect']
        self.in_channels = model_cfg['in_channels']
        self.out_channels = model_cfg['out_channels']
        self.width = model_cfg['width']
        self.head_layer = model_cfg['head']
        self.first_channels = self.width if self.head_layer else self.in_channels
        if self.head_layer:
            self.head = nn.Conv2d(self.in_channels, self.width, kernel_size=3, stride=1, padding=1)

        # Now we fix the block size as 4
        self.encoder_1 = BlockType(self.first_channels, self.width, res_connect=self.res_connect)
        self.pool_1 = nn.Conv2d(self.width, self.width, kernel_size=4, stride=2, padding=1)
        self.encoder_2 = BlockType(self.width, self.width*2, res_connect=self.res_connect)
        self.pool_2 = nn.Conv2d(self.width*2, self.width*2, kernel_size=4, stride=2, padding=1)
        self.encoder_3 = BlockType(self.width*2, self.width*4, res_connect=self.res_connect)
        self.pool_3 = nn.Conv2d(self.width*4, self.width*4, kernel_size=4, stride=2, padding=1)
        self.encoder_4 = BlockType(self.width*4, self.width*8, res_connect=self.res_connect)
        self.pool_4 = nn.Conv2d(self.width*8, self.width*8, kernel_size=4, stride=2, padding=1)

        self.mid = ResBlock(self.width*8, self.width*16, res_connect=self.res_connect)

        self.up_1 = nn.ConvTranspose2d(self.width*16, self.width*8, 2, stride=2)
        self.decoder_1 = BlockType(self.width*16, self.width*8, res_connect=self.res_connect)
        self.up_2 = nn.ConvTranspose2d(self.width*8, self.width*4, 2, stride=2)
        self.decoder_2 = BlockType(self.width*8, self.width*4, res_connect=self.res_connect)
        self.up_3 = nn.ConvTranspose2d(self.width*4, self.width*2, 2, stride=2)
        self.decoder_3 = BlockType(self.width*4, self.width*2, res_connect=self.res_connect)
        self.up_4 = nn.ConvTranspose2d(self.width*2, self.width, 2, stride=2)
        self.decoder_4 = BlockType(self.width*2, self.width, res_connect=self.res_connect)

        self.tail = nn.Conv2d(self.width, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if self.head_layer:
            head = self.head(x)
        else:
            head = x

        conv1 = self.encoder_1(head)
        pool1 = self.pool_1(conv1)

        conv2 = self.encoder_2(pool1)
        pool2 = self.pool_2(conv2)

        conv3 = self.encoder_3(pool2)
        pool3 = self.pool_3(conv3)

        conv4 = self.encoder_4(pool3)
        pool4 = self.pool_4(conv4)

        conv5 = self.mid(pool4)

        up6 = self.up_1(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.decoder_1(up6)

        up7 = self.up_2(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.decoder_2(up7)

        up8 = self.up_3(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.decoder_3(up8)

        up9 = self.up_4(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.decoder_4(up9)

        conv10 = self.tail(conv9)
        out = x + conv10
        return out

# --------------------------- Model Instantiation --------------------------- #
def resunet(model_cfg, model_path: str=None, device=None):
    model = BlockUNet(model_cfg)
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        print(f'load pretrained checkpoint from: {model_path}')
        model.load_state_dict(state_dict)
    return model

def dconvunet(model_cfg, model_path: str=None, device=None):
    model = BlockUNet(model_cfg, BlockType=DConvBlock)
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        print(f'load pretrained checkpoint from: {model_path}')
        model.load_state_dict(state_dict)
    return model


