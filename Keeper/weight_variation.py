# 这个脚本是量化版本的，验证用。

import os
import glob
import ipdb
import time
import math
import random
import shutil
import argparse
import warnings
import numpy as np

import torch
import torch.nn as nn
import utils

import models
import datasets
from utils.metric import calc_SSIM, calc_PSNR
from utils.quant_utils import get_qscheme, get_qconfig_dict, prepare_custom_config_dict, calibrate

from mqbench.convert_deploy import convert_deploy
from mqbench.weight_variation import inject_weight_variation
from mqbench.prepare_by_platform import prepare_qat_fx_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization, disable_all

model_names = sorted(name for name in models.__dict__ \
                if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

### -------------------- Parser Zone  -------------------- ###
parser = argparse.ArgumentParser("☆ Welcome to the ZOO of LLCV - quant ver!☆")
parser.add_argument('--arch', metavar='ARCH', default='lsid')
parser.add_argument('--root', type=str, default='./datasets/SIDD/SIDD_patches/', help='root location of the data corpus')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
#parser.add_argument('--steps', type=int, default=100, help='steps of each epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--save_name', type=str, default='HE_valid', help='experiment name')
parser.add_argument('--save_path', type=str, default='./checkpoints', help='parent path for saved experiments')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--patch_size', type=int, default=512, help='patch size')
parser.add_argument('--gpu', type=str, default='1', help='gpu device ids')
parser.add_argument('--seed', type=int, default=99, help='seed for initializing training')
parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: None)')
parser.add_argument('--resume', type=str, default=None, 
                    help='checkpoint path of previous model, loading for evaluation or retrain')
parser.add_argument('--eager_test', dest='eager_test', action='store_true', 
                    help='debug only. test per 10 epochs during training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--not_save', dest='save_flag', action='store_false',
                    help='if specified, logs won\'t be saved')

def main():
    args = parser.parse_args()
    if args.save_flag:
        # create dir for saving results
        args.save_name = '{}-{}-{}'.format(args.save_name, 'test' if args.evaluate else 'train',
                                            time.strftime("%Y%m%d-%H%M%S"))
        args.save_path = os.path.join(args.save_path, args.save_name)
        # scripts & configurations to be saved
        save_list = ['qmain.py', 'utils/optimizer.py', 'models/unet_denoise_.py']
        utils.create_exp_dir(args.save_path, scripts_to_save=save_list)

    # get info logger
    logging = utils.get_logger(args)

    # set up device.
    if args.gpu and torch.cuda.is_available():
        args.gpu_flag = True
        device = torch.device('cuda')
        gpus = [int(d) for d in args.gpu.split(',')]
        torch.cuda.set_device(gpus[0]) # currently only training & inference on single card is supported.
        logging.info("Using GPU(s). Available gpu count: {}".format(torch.cuda.device_count()))
    else:
        device = torch.device('cpu')
        logging.info("\033[1;3mWARNING: Using CPU!\033[0m")

    # set random seed
    if args.seed:
        # cpu seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
        logging.info("Setting Random Seed {}".format(args.seed))

    # set up model & optimizer & dataset 
    model = models.__dict__[args.arch](inchannel=3, outchannel=3).cuda()
    # model = models.lsid(inchannel=4, outchannel=4)

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    criterion = nn.L1Loss()

    # SIDD only
    Loader_Settings = {
        'num_workers': args.workers,
        'pin_memory':  True,
        'batch_size':  args.batch_size}

    train_data = datasets.SIDD_Medium_sRGB_Train_DataLoader(os.path.join(args.root, 'train'), 96000, 256, True)
    val_data = datasets.SIDD_sRGB_Val_DataLoader(os.path.join(args.root, 'val'))
    test_data = datasets.SIDD_sRGB_mat_Test_DataLoader(os.path.join(args.root, 'test'))
    cali_data = torch.utils.data.Subset(train_data, indices=torch.arange(9600))

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **Loader_Settings)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **Loader_Settings)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **Loader_Settings)
    cali_loader = torch.utils.data.DataLoader(cali_data, shuffle=False, **Loader_Settings)


    start_epoch = 0
    best_psnr = 0
    best_psnr_epoch = 0
    best_ssim = 0
    best_ssim_epoch = 0
    best_loss = float('inf')
    best_loss_epoch = 0
 
    # quantize model
    get_qconfig_dict('io',       w_qscheme=get_qscheme(),      a_qscheme=get_qscheme(symmetry=False, per_channel=False))
    get_qconfig_dict('default',  w_qscheme=get_qscheme(bit=8), a_qscheme=get_qscheme(bit=8, per_channel=False))
    model = prepare_qat_fx_by_platform(model, BackendType.Custom, prepare_custom_config_dict)
    if args.gpu_flag:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        logging.info("Using CPU. This will be slow.")

    # Resume a model if provided
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Loading checkpoint '{}'".format(args.resume))
            if not args.gpu:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch=best_psnr_epoch=best_ssim_epoch=best_loss_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info("No checkpoint found at '{}', please check.".format(args.resume))
            return

    # 2022/2/12 update - carry a noise injection exp.
    bit_dict = {}
    import ipdb; ipdb.set_trace()
    # model = inject_weight_variation(model, 8, bit_dict, variation=1./32.)

    if not args.evaluate:
        # If this is not single evaluation, should calibrate the model first.
        enable_calibration(model)
        calibrate(cali_loader, model, logging, args)
        enable_quantization(model)
    
    elif args.evaluate:
        assert args.resume, "You should provide a checkpoint through args.resume."
        from mqbench.convert_deploy import convert_merge_bn
        convert_merge_bn(model.eval())
        psnr, ssim, loss = infer(model, test_loader, criterion, args, logging)
        logging.info("Average PSNR {}, Average SSIM {}, Average Loss {}"
                     .format(psnr, ssim, loss))
        return

    # training progress
    for epoch in range(0 if not start_epoch else start_epoch, args.epochs): 
        # train one epoch
        logging.info('Epoch [%d/%d]  lr: %e', epoch+1, args.epochs,
                     optimizer.state_dict()['param_groups'][0]['lr'])#scheduler.get_last_lr()[0])
        logging.info('<-Training Phase->')
        train(model, train_loader, criterion, optimizer, args, logging)

        # validate last epoch
        logging.info('<-Validating Phase->')
        psnr, ssim, loss = infer(model, val_loader, criterion, args, logging)

        # test one epoch if it's under eager mode
        if args.eager_test and epoch % 10 == 0:
            logging.info('<-Testing Phase->')
            psnr_test, ssim_test, loss_test = infer(model, test_loader, criterion, args, logging)
            logging.info('Test - PSNR:%4f SSIM:%4f Loss:%4f', psnr_test, ssim_test, loss_test)

        # model save
        best_names = []
        if psnr > best_psnr and not math.isinf(psnr):
            best_names.append('best_psnr.pth.tar')
            best_psnr_epoch = epoch + 1
            best_psnr = psnr
        if ssim > best_ssim:
            best_names.append('best_ssim.pth.tar')
            best_ssim_epoch = epoch + 1
            best_ssim = ssim
        if loss < best_loss:
            best_names.append('best_loss.pth.tar')
            best_loss_epoch = epoch + 1
            best_loss = loss
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'psnr': psnr,
            'ssim': ssim,
            'loss': loss,
            'optimizer': optimizer.state_dict()}, best_names, args.save_path) 

        # scheduler.step()
        adjust_learning_rate(optimizer, epoch, args)

        logging.info('PSNR:%4f SSIM:%4f Loss:%4f / Best_PSNR:%4f Best_SSIM:%4f Best_Loss:%4f', 
                     psnr, ssim, loss, best_psnr, best_ssim, best_loss)
    logging.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)', 
                 best_loss, best_loss_epoch, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch)

def train(model, train_loader, criterion, optimizer, args, logging): 
    Loss = utils.AverageMeter('Loss')
    PSNR = utils.AverageMeter('PSNR')
    SSIM = utils.AverageMeter('SSIM')
    Batch_time = utils.AverageMeter('batch time')

    # model state
    model.train()

    # timer
    end = time.time()

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            psnr = calc_PSNR(outputs, targets)
            ssim = calc_SSIM(torch.clamp(outputs,0,1), targets)

        # display
        Loss.update(loss.item(), inputs.size(0))
        Batch_time.update(time.time() - end)
        PSNR.update(psnr, inputs.size(0))
        SSIM.update(ssim, inputs.size(0))
        end = time.time()

        if args.print_freq is not None and step % args.print_freq == 0:
            logging.info('batch [{0}/{1}] \t'
                         'Time {Batch_time.val:.3f} ({Batch_time.avg:.3f})\t'
                         'Loss {Loss.val:.3f} ({Loss.avg:.3f})\t'
                         'PSNR {PSNR.val:.3f} ({PSNR.avg:.3f})\t'
                         'SSIM {SSIM.val:.3f} ({SSIM.avg:.3f})'
                         .format(step, len(train_loader), Batch_time=Batch_time, Loss=Loss,
                                 PSNR=PSNR, SSIM=SSIM))

def infer(model, val_loader, criterion, args, logging):
    Loss = utils.AverageMeter('Loss')
    Batch_time = utils.AverageMeter('batch time')
    PSNR = utils.AverageMeter('PSNR')
    SSIM = utils.AverageMeter('SSIM')
    
    # timer
    end = time.time()

    model.eval()
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            ssim = calc_SSIM(torch.clamp(outputs,0,1), targets)
            psnr = calc_PSNR(outputs, targets)
            n = inputs.size(0)
            PSNR.update(psnr, n)
            SSIM.update(ssim, n)
            Loss.update(loss.item(), n)
            Batch_time.update(time.time() - end)
            end = time.time()

            if args.print_freq is not None and batch % args.print_freq == 0:
                logging.info('batch [{0}/{1}]  \t'
                             'Time {Batch_time.val:.3f} ({Batch_time.avg:.3f})\t'
                             'Loss {Loss.val:.3f} ({Loss.avg:.3f})\t' 
                             'PSNR {PSNR.val:.3f} ({PSNR.avg:.3f})\t'
                             'SSIM {SSIM.val:.3f} ({SSIM.avg:.3f})\t'
                             .format(batch, len(val_loader), Batch_time=Batch_time, Loss=Loss,
                                     PSNR=PSNR, SSIM=SSIM))
 
    return PSNR.avg, SSIM.avg, Loss.avg

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 40))
    if epoch <= 100:
        lr = 1e-4
    elif epoch <= 180:
        lr = 5e-5
    else:
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
