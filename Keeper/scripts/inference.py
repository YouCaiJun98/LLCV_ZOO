# This script should run under ./Keeper
import os
import sys
import cv2
import logging
import argparse
import numpy as np
from skimage import img_as_ubyte
from collections import OrderedDict

import torch
import torch.nn as nn

import models
import datasets
from utils.utils import calc_PSNR
from utils.utils import get_logger
from utils.utils import AverageMeter
from utils.utils import ProgressMeter
from utils.utils import get_state_dict
from utils.utils import create_exp_dir
from utils.quant_utils import get_qscheme
from utils.quant_utils import get_qconfig_dict
from mqbench.prepare_by_platform import prepare_qat_fx_by_platform, BackendType

model_names = sorted(name for name in models.__dict__ \
                if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
# import ipdb; ipdb.set_trace()
### -------------------- Paser Zone  -------------------- ###
parser = argparse.ArgumentParser(description='☆ LLCV Zoo Openup Daze ☆')
parser.add_argument('--arch', metavar='ARCH', default='lsid')
parser.add_argument('--dataset', metavar='DIR',
                    help='path to dataset', default='./datasets/SIDD/SIDD_patches')
parser.add_argument('--resume', type=str, default='../../MQBench/application/imagenet_example/checkpoint/' + \
                    'denoise/denoise_48_fix/model_best.pth.tar')
parser.add_argument('--save_path', metavar='DIR',
                    help='path to save output images and loggings', default='./test')
parser.add_argument('--quantized', action='store_true',
                    help='use a quantized model to output an img.')
parser.add_argument('--save_img', action='store_true',
                    help='save the processed imgs to dir.')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--patch_size', default=256, type=int, metavar='N',
                    help='choose to crop raw image as patch size')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')

### --------------------- Env Setup  -------------------- ###
args = parser.parse_args()
args.evaluate = True
create_exp_dir(args.save_path)
logger = get_logger(args)
torch.cuda.set_device(args.gpu)
device = torch.device('cuda')

### -------------------- Model Setup  ------------------- ###
assert args.arch in model_names, 'Should specify a known model.' 
model = models.__dict__[args.arch]().cuda()
if args.quantized:
    prepare_custom_config_dict = {}
    prepare_custom_config_dict = get_qconfig_dict(prepare_custom_config_dict, 'io', w_qscheme=get_qscheme(), w_fakequantize='FixedFakeQuantize', a_fakequantize='FixedFakeQuantize', a_qscheme=get_qscheme(symmetry=False, per_channel=True))
    prepare_custom_config_dict = get_qconfig_dict(prepare_custom_config_dict, 'default', w_qscheme=get_qscheme(bit=4), w_fakequantize='FixedFakeQuantize', a_fakequantize='FixedFakeQuantize', a_qscheme=get_qscheme(bit=4))
    model = prepare_qat_fx_by_platform(model, BackendType.Custom, prepare_custom_config_dict).cuda()

criterion = nn.L1Loss().cuda(args.gpu)


### ----------------- Load Checkpoints  ----------------- ###
loc = 'cuda:{}'.format(args.gpu)
checkpoint = torch.load(args.resume, map_location=loc)
state_dict = get_state_dict(checkpoint)
model.load_state_dict(state_dict)


### ----------------- Setup DataLoader  ----------------- ###
test_set = datasets.SIDD_sRGB_Val_DataLoader(os.path.join(args.dataset, 'val'))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)


### ---------------- Validation Mainbody  --------------- ###
batch_time = AverageMeter('Time', ':6.3f')
losses = AverageMeter('Loss', ':.4e')
psnr = AverageMeter('PSNR', ':2.3f')
progress = ProgressMeter(
    len(test_loader),
    [batch_time, losses, psnr],
    prefix='Inference: ', logger=logger)
model.eval()
# import ipdb; ipdb.set_trace()
with torch.no_grad():
    for i, (noisy_imgs, clean_imgs, filename) in enumerate(test_loader):
        noisy_imgs, clean_imgs = noisy_imgs.cuda(), clean_imgs.cuda()

        restored_imgs = model(noisy_imgs)
        restored_imgs = restored_imgs[0]
        clean_imgs = clean_imgs[:, :, :restored_imgs.size(2), :restored_imgs.size(3)]
        loss = criterion(restored_imgs, clean_imgs)

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while validating')

        # measure accuracy and record loss
        losses.update(loss.item(), restored_imgs.size(0))
        psnr.update(calc_PSNR(restored_imgs, clean_imgs, 1.0), restored_imgs.size(0))

        if args.save_img:
            # note that this is for batch size = 1 case
            img = torch.clamp(restored_imgs,0,1).cpu().permute(0, 2, 3, 1).squeeze(0)
            img_dir = os.path.join(args.save_path, '{}.png'.format(filename[0]))
            img = img_as_ubyte(img)
            cv2.imwrite(img_dir, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if i % args.print_freq == 0:
            progress.display(i, filename[0])
    progress.display(i+1, filename[0])

import ipdb;ipdb.set_trace()


