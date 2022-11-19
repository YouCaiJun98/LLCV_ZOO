import os
import sys
import glob
import time
import math
import yaml
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
import models
# from model.unet import make_model
import datasets
from utils.metric import calc_SSIM, calc_PSNR
from utils.optimizer import adjust_learning_rate
# from Utils.warmup_scheduler import GradualWarmupScheduler

### -------------------- Parser Zone  -------------------- ###
parser = argparse.ArgumentParser("☆ Welcome to the ZOO of LLCV ☆")
parser.add_argument('-a', '--arch', metavar='ARCH', default='lsid')
parser.add_argument('--root', type=str, default='./datasets/Sony', help='root location of the data corpus')
parser.add_argument('--save_name', type=str, default='Patch_Saving', help='experiment name')
parser.add_argument('--save_path', type=str, default='./saved_patches', help='parent path for saved experiments')
parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: None)')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint path of previous model, loading for evaluation or retrain')
parser.add_argument('-c', '--configuration', required=True, help='model & train/validate settings')
parser.add_argument('-d', '--debug', dest='save_flag', action='store_false',
                    help='if specified, logs won\'t be saved')
parser.add_argument('--local_rank', default=0, type=int,
                    help='Local rank for each gpu under distribution environment.')
parser.add_argument('--save_img', action='store_true',)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main():
    # get model dicts
    # model_names = sorted(name for name in models.__dict__ \
    #             if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
    args = parser.parse_args()
    # To distinguish master and slave under distributed training scene.
    args.slave = True if args.local_rank != 0 else False
    if args.save_flag:
        # create dir for saving results
        args.save_name = '{}-{}'.format(args.save_name, time.strftime("%Y%m%d-%H%M%S"))
        args.save_path = os.path.join(args.save_path, args.save_name)
        # scripts & configurations to be saved
        # save_list = ['model/unet.py'] + [__file__] + [args.configuration]
        save_list = []
        if args.local_rank == 0:
            utils.create_exp_dir(args.save_path, scripts_to_save=save_list)
        if args.save_img:
            args.img_dir = os.path.join(args.save_path, 'saved_patches')
            if not os.path.exists(args.img_dir):
                os.makedirs(args.img_dir)

    # parse configurations
    with open(args.configuration, 'r') as rf:
        cfg = yaml.load(rf, Loader=yaml.FullLoader)
        train_cfg = cfg['training_settings']
        model_cfg = cfg['model_settings']
    gpu = str(train_cfg['gpu'])
    epochs = train_cfg['epochs']
    workers = train_cfg['workers']
    init_lr = train_cfg['init_lr']
    lr_schedule = train_cfg['lr_schedule']
    grad_clip = train_cfg['grad_clip']
    batch_size = train_cfg['batch_size']
    patch_size = train_cfg['patch_size']
    seed = train_cfg['seed'] if train_cfg['seed'] else None
    # get info logger
    logging = utils.get_logger(args)

    start_epoch = 0
    best_psnr = 0
    best_psnr_epoch = 0
    best_ssim = 0
    best_ssim_epoch = 0
    best_loss = float('inf')
    best_loss_epoch = 0

    # set up device.
    if gpu:
        args.gpu_flag = True
        args.multi_cards = False if len(gpu) == 1 else True
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        if len(gpu) == 1:
            device = torch.device('cuda')
            torch.cuda.set_device(0) # single card case.
        elif len(gpu) > 1:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl')
            device = torch.device('cuda', args.local_rank)
            world_size = torch.distributed.get_world_size() # batch size in cfg file refers to the total size.
            args.rank = torch.distributed.get_rank()
        args.device = device
        logging.info("Using GPU {}. Available gpu count: {}".format(gpu, torch.cuda.device_count()))
    else:
        args.gpu_flag = False
        device = torch.device('cpu')
        args.device = device
        logging.info("\033[1;3mWARNING: Using CPU!\033[0m")

    # set random seed
    if seed:
        # cpu seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
        logging.info("Setting Random Seed {}".format(seed))

    # set up model & optimizer & dataset 
    criterion = nn.L1Loss()

    # TODO says SXS: Build Your Model Here and Load Your Checkpoint.
    # model = models.__dict__[args.arch](model_cfg)
    model = models.lsid(model_cfg)
    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), init_lr)

    # Resume a model if provided, only master should load ckpt.
    if args.resume and not args.slave:
        if os.path.isfile(args.resume):
            logging.info("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            if 'epoch' in checkpoint:
                start_epoch=best_psnr_epoch=best_ssim_epoch=best_loss_epoch = checkpoint['epoch']

            if 'total_params' in checkpoint['state_dict']:
                checkpoint['state_dict'].pop('total_params')
                checkpoint['state_dict'].pop('total_ops')
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info("No checkpoint found at '{}', please check.".format(args.resume))
            return

    if args.gpu_flag:
        model.to(device)
        criterion.to(device)
        if args.multi_cards:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        logging.info("Using CPU. This will be slow.")

    Loader_Settings = {
        'num_workers': workers,
        'pin_memory':  True,
        'batch_size':  batch_size}


    # SID-Sony only
    img_list_files = ['./datasets/Sony/Sony_train_png.txt',
                      './datasets/Sony/Sony_val_png.txt',
                      './datasets/Sony/Sony_test_png.txt']
    train_data = datasets.SID_Sony(args.root, img_list_files[0], patch_size=patch_size,  data_aug=True,  stage_in='raw', stage_out='sRGB', gt_png=True, return_name=True)
    val_data   = datasets.SID_Sony(args.root, img_list_files[1], patch_size=128, data_aug=False, stage_in='raw', stage_out='sRGB', gt_png=True, return_name=True)
    test_data  = datasets.SID_Sony(args.root, img_list_files[2], patch_size=None, data_aug=False, stage_in='raw', stage_out='sRGB', gt_png=True, return_name=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=workers, pin_memory=True)

    '''
    # SIDD only
    train_data = datasets.SIDD_sRGB_Train_DataLoader(os.path.join(args.root, 'train'), 96000, 256, True)
    val_data = datasets.SIDD_sRGB_Val_DataLoader(os.path.join(args.root, 'val'), return_name=True)
    test_data  = datasets.SIDD_sRGB_mat_Test_DataLoader(os.path.join(args.root, 'test'))

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **Loader_Settings)
    # says SXS: fix test batch size as 1.
    Loader_Settings['batch_size'] = 1
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **Loader_Settings)
    test_loader  = torch.utils.data.DataLoader(test_data,  shuffle=False, **Loader_Settings)
    '''
    '''
    # GoPro only
    train_data = datasets.GoPro_sRGB_Train_DataSet(os.path.join(args.root, 'train'), 256)
    test_data = datasets.GoPro_sRGB_Test_DataSet(os.path.join(args.root, 'test'))
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **Loader_Settings)
    Loader_Settings['batch_size'] = 4
    val_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **Loader_Settings)
    test_loader  = torch.utils.data.DataLoader(test_data,  shuffle=False, **Loader_Settings)
    '''

    if args.multi_cards:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        Loader_Settings['batch_size'] = batch_size // world_size
        assert Loader_Settings['batch_size'] * world_size == batch_size, 'batch size is not divisible.'
        train_loader = torch.utils.data.DataLoader(train_data, **Loader_Settings, sampler=train_sampler)

    # Clear these out

    assert args.resume, "You should provide a checkpoint through args.resume."
    # says SXS: because val_loader of SIDD can provide with filename.
    dataloader = val_loader if 'SIDD' in args.root else test_loader
    psnr, ssim, loss = infer(model, dataloader, criterion, args, logging)
    logging.info("Average PSNR {}, Average SSIM {}, Average Loss {}"
                    .format(psnr, ssim, loss))
    return

def infer(model, val_loader, criterion, args, logging):
    Loss = utils.AverageMeter('Loss')
    Batch_time = utils.AverageMeter('batch time')
    PSNR = utils.AverageMeter('PSNR')
    SSIM = utils.AverageMeter('SSIM')

    # timer
    end = time.time()

    model.eval()
    with torch.no_grad():
        for batch, (inputs, targets, fn) in enumerate(val_loader):
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            ssim = calc_SSIM(torch.clamp(outputs,0,1), targets)
            psnr = calc_PSNR(outputs, targets)
            PSNR.update(psnr, batch_size)
            SSIM.update(ssim, batch_size)
            Loss.update(loss.item(), batch_size)
            Batch_time.update(time.time() - end)
            end = time.time()

            if args.save_img:
                img_name = fn[0] + '-' + str(psnr)[:2] + '_' + str(psnr)[3:5] + '.png'
                img = outputs.cpu().squeeze()
                save_image(img, os.path.join(args.img_dir, img_name))


            if args.print_freq is not None and batch % args.print_freq == 0:
                logging.info('batch [{0}/{1}]  \t'
                             'Time {Batch_time.val:.3f} ({Batch_time.avg:.3f})\t'
                             'Loss {Loss.val:.3f} ({Loss.avg:.3f})\t' 
                             'PSNR {PSNR.val:.3f} ({PSNR.avg:.3f})\t'
                             'SSIM {SSIM.val:.3f} ({SSIM.avg:.3f})\t'
                             .format(batch, len(val_loader), Batch_time=Batch_time, Loss=Loss,
                                     PSNR=PSNR, SSIM=SSIM))

    return PSNR.avg, SSIM.avg, Loss.avg


if __name__ == '__main__':
    main()
