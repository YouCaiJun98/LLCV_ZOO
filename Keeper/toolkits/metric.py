'''
Code: https://github.com/Po-Hsun-Su/pytorch-ssim
'''
import numpy as np
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .metric_utils import rgb2ycbcr_pt


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

#  (B, C, H, W)
def calc_SSIM(img1, img2, window_size = 11, size_average = True,
              crop_border=0, test_y_channel=False):
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'

    (_, channel, _, _) = img1.size()
    if test_y_channel:
        channel = 1
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    if crop_border != 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img1 = rgb2ycbcr_pt(img1, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    return _ssim(img1, img2, window, window_size, channel, size_average)


from typing import Union
def calc_PSNR(inputs:torch.Tensor, targets:torch.Tensor,
              data_range:Union[int, float]=1.0,
              crop_border:int=0, test_y_channel:bool=False) -> float:
    '''
    Args:
        inputs - the input tensor, should be of shape (N, C, H, W).
        targets - the target tensor, should be of shape (N, C, H, W).
        data_range - the data range of the given tensors, should be in ['255', '1.0'].
        reduction - the method to handle batched results. should be in ['none', 'sum', 'mean'].
    Returns:
        PSNR - the calculated results (mean value of the input batched sample).
    Reference:
        https://github.com/photosynthesis-team/piq/blob/master/piq/psnr.py
    '''
    assert inputs.shape == targets.shape, f'Image shapes are different: {inputs.shape}, {targets.shape}.'

    if crop_border != 0:
        inputs = inputs[:, :, crop_border:-crop_border, crop_border:-crop_border]
        targets = targets[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        inputs = rgb2ycbcr_pt(inputs, y_only=True)
        targets = rgb2ycbcr_pt(targets, y_only=True)

    # Constant for numerical stability, could guarantee accuracy in .5f
    eps = 1e-10
    inputs = torch.clamp(inputs, 0, float(data_range))
    inputs, targets = inputs/float(data_range), targets/float(data_range)
    MSE = torch.mean((inputs - targets) ** 2, dim=[1, 2, 3])
    PSNR: torch.Tensor = - 10 * torch.log10(MSE + eps)
    return PSNR.mean(dim=0).item()


