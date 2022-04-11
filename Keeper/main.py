import os
import sys
import glob
import time
import math
import random
import argparse
import numpy as np

import torch
import torch.nn as nn


import utils
import models
from models.unet import lsid
import datasets
from utils.metric import calc_SSIM, calc_PSNR

### -------------------- Parser Zone  -------------------- ###
parser = argparse.ArgumentParser("☆ Welcome to the ZOO of LLCV ☆")
parser.add_argument('--arch', metavar='ARCH', default='lsid')
parser.add_argument('--root', type=str, default='./datasets/SIDD/SIDD_patches/', help='root location of the data corpus')
parser.add_argument('--epochs', type=int, default=80, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1.e-4, help='init learning rate')
parser.add_argument('--save_name', type=str, default='HE_valid', help='experiment name')
parser.add_argument('--save_path', type=str, default='./checkpoints', help='parent path for saved experiments')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--patch_size', type=int, default=512, help='patch size')
parser.add_argument('--gpu', type=str, help='gpu device ids')
parser.add_argument('--seed', type=int, default=99, help='seed for initializing training')
parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: None)')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint path of previous model, loading for evaluation or retrain')
parser.add_argument('--eager_test', dest='eager_test', action='store_true',
                    help='debug only. test per 10 epochs during training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-d', '--debug', dest='save_flag', action='store_false',
                    help='if specified, logs won\'t be saved')

def main():
    # get model dicts
    model_names = sorted(name for name in models.__dict__ \
                if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
    args = parser.parse_args()
    if args.save_flag:
        # create dir for saving results
        args.save_name = '{}-{}-{}'.format(args.save_name, 'test' if args.evaluate else 'train',
                                            time.strftime("%Y%m%d-%H%M%S"))
        args.save_path = os.path.join(args.save_path, args.save_name)
        # scripts & configurations to be saved
        save_list = ['main.py', 'models/unet.py']
        utils.create_exp_dir(args.save_path, scripts_to_save=save_list)

    # get info logger
    logging = utils.get_logger(args)

    # set up device.
    if args.gpu and torch.cuda.is_available():
        args.gpu_flag = True
        device = torch.device('cuda')
        gpus = [int(d) for d in args.gpu.split(',')]
        torch.cuda.set_device(gpus[0]) # currently only single card is supported
        logging.info("Using GPU {}. Available gpu count: {}".format(gpus[0], torch.cuda.device_count()))
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
    criterion = nn.L1Loss()

    model = models.__dict__[args.arch](inchannel=3, outchannel=3, base_ch=5)
    logging.info(model)

    if args.gpu_flag:
        model.cuda()
        criterion.cuda()
    else:
        logging.info("Using CPU. This will be slow.")

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    '''
    # SID-Sony only
    img_list_files = ['./datasets/Sony/Sony_train_list.txt',
                      './datasets/Sony/Sony_val_list.txt',
                      './datasets/Sony/Sony_test_list.txt']
    train_data = datasets.SID_Sony(args.dataset, img_list_files[0], patch_size=args.patch_size,  data_aug=True,  stage_in='raw', stage_out='raw')
    val_data   = datasets.SID_Sony(args.dataset, img_list_files[1], patch_size=None, data_aug=False, stage_in='raw', stage_out='raw')
    test_data  = datasets.SID_Sony(args.dataset, img_list_files[2], patch_size=None, data_aug=False, stage_in='raw', stage_out='raw')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=args.workers, pin_memory=True)
    '''

    # SIDD only
    Loader_Settings = {
        'num_workers': args.workers,
        'pin_memory':  True,
        'batch_size':  args.batch_size}
    train_data = datasets.SIDD_sRGB_Train_DataLoader(os.path.join(args.root, 'train'), 96000, 256, True)
    val_data = datasets.SIDD_sRGB_Val_DataLoader(os.path.join(args.root, 'val'))
    test_data  = datasets.SIDD_sRGB_mat_Test_DataLoader(os.path.join(args.root, 'test'))
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **Loader_Settings)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **Loader_Settings)
    test_loader  = torch.utils.data.DataLoader(test_data,  shuffle=False, **Loader_Settings)

    start_epoch = 0
    best_psnr = 0
    best_psnr_epoch = 0
    best_ssim = 0
    best_ssim_epoch = 0
    best_loss = float('inf')
    best_loss_epoch = 0

    # Resume a model if provided
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch=best_psnr_epoch=best_ssim_epoch=best_loss_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info("No checkpoint found at '{}', please check.".format(args.resume))
            return

    # Clear these out

    import ipdb; ipdb.set_trace()
    from thop import profile
    dummy_input = [torch.randn(1,3,256,256).cuda()]
    flops, params = profile(model, inputs=dummy_input, verbose=True)


    if args.evaluate:
        # assert args.resume, "You should provide a checkpoint through args.resume."
        psnr, ssim, loss = infer(model, test_loader, criterion, args, logging)
        logging.info("Average PSNR {}, Average SSIM {}, Average Loss {}"
                     .format(psnr, ssim, loss))
        return

    # training progress
    for epoch in range(0 if not start_epoch else start_epoch, args.epochs):

        # train one epoch
        logging.info('Epoch [%d/%d]  lr: %e', epoch+1, args.epochs,
                     optimizer.state_dict()['param_groups'][0]['lr'])
        logging.info('<-Training Phase->')

        # train one epoch
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
        if args.save_flag:
            utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'psnr': psnr,
            'ssim': ssim,
            'loss': loss,
            'optimizer': optimizer.state_dict()}, best_names, args.save_path)

        adjust_learning_rate(optimizer, epoch)

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

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def adjust_learning_rate(optimizer, epoch):
    """Sets multi-step LR scheduler."""
    if epoch <= 20:
        lr = 1e-4
    elif epoch <= 40:
        lr = 5e-5
    elif epoch <= 60:
        lr = 2.5e-5
    else:
        lr = 1.25e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
