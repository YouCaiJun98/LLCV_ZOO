import os
import sys
import glob
import time
import math
import yaml
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn

import utils
import models
import datasets
from utils.metric import calc_SSIM, calc_PSNR
from utils.optimizer import adjust_learning_rate

### -------------------- Parser Zone  -------------------- ###
parser = argparse.ArgumentParser("☆ Welcome to the ZOO of LLCV ☆")
parser.add_argument('-a', '--arch', metavar='ARCH', default='lsid')
parser.add_argument('--root', type=str, default='./datasets/Sony', help='root location of the data corpus')
parser.add_argument('--save_name', type=str, default='HE_valid', help='experiment name')
parser.add_argument('--save_path', type=str, default='./checkpoints', help='parent path for saved experiments')
parser.add_argument('--gpu', type=str, help='gpu device ids')
parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: None)')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint path of previous model, loading for evaluation or retrain')
parser.add_argument('--eager_test', dest='eager_test', action='store_true',
                    help='debug only. test per 10 epochs during training')
parser.add_argument('-c', '--configuration', required=True, help='model & train/validate settings')
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
        save_list = ['models/unet.py'] + [__file__] + [args.configuration]
        utils.create_exp_dir(args.save_path, scripts_to_save=save_list)

    # parse configurations
    with open(args.configuration, 'r') as rf:
        cfg = yaml.load(rf, Loader=yaml.FullLoader)
        train_cfg = cfg['training_settings']
        model_cfg = cfg['model_settings']
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

    # set up device.
    if args.gpu and torch.cuda.is_available():
        args.gpu_flag = True
        device = torch.device('cuda')
        gpus = [int(d) for d in args.gpu.split(',')]
        torch.cuda.set_device(gpus[0]) # currently only single card is supported
        logging.info("Using GPU {}. Available gpu count: {}".format(gpus[0], torch.cuda.device_count()))
    else:
        args.gpu_flag = False
        device = torch.device('cpu')
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

    model = models.__dict__[args.arch](model_cfg)
    logging.info(model)

    if args.gpu_flag:
        model.cuda()
        criterion.cuda()
    else:
        logging.info("Using CPU. This will be slow.")

    optimizer = torch.optim.Adam(model.parameters(), init_lr)

    # SID-Sony only
    img_list_files = ['./datasets/Sony/Sony_train_png.txt',
                      './datasets/Sony/Sony_val_png.txt',
                      './datasets/Sony/Sony_test_png.txt']
    train_data = datasets.SID_Sony(args.root, img_list_files[0], patch_size=patch_size,  data_aug=True,  stage_in='raw', stage_out='sRGB', gt_png=True)
    val_data   = datasets.SID_Sony(args.root, img_list_files[1], patch_size=None, data_aug=False, stage_in='raw', stage_out='sRGB', gt_png=True)
    test_data  = datasets.SID_Sony(args.root, img_list_files[2], patch_size=None, data_aug=False, stage_in='raw', stage_out='sRGB', gt_png=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=workers, pin_memory=True)

    '''
    # SIDD only
    Loader_Settings = {
        'num_workers': workers,
        'pin_memory':  True,
        'batch_size':  batch_size}
    train_data = datasets.SIDD_sRGB_Train_DataLoader(os.path.join(args.root, 'train'), 96000, 256, True)
    val_data = datasets.SIDD_sRGB_Val_DataLoader(os.path.join(args.root, 'val'))
    test_data  = datasets.SIDD_sRGB_mat_Test_DataLoader(os.path.join(args.root, 'test'))
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **Loader_Settings)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **Loader_Settings)
    test_loader  = torch.utils.data.DataLoader(test_data,  shuffle=False, **Loader_Settings)
    '''

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
            if args.resume.endswith('pkl'):
                with open(args.resume, 'rb') as f:
                    weights = pickle.load(f, encoding='latin1')
                own_state = model.state_dict()
                for name, param in weights.items():
                    if name in own_state:
                        try:
                            own_state[name].copy_(torch.from_numpy(param))
                        except Exception:
                            raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                               'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
                    else:
                        raise KeyError('unexpected key "{}" in state_dict'.format(name))
            else:
                checkpoint = torch.load(args.resume, map_location=device)
                start_epoch=best_psnr_epoch=best_ssim_epoch=best_loss_epoch = checkpoint['epoch']

                if 'total_params' in checkpoint['state_dict']:
                    checkpoint['state_dict'].pop('total_params')
                    checkpoint['state_dict'].pop('total_ops')

                model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info("No checkpoint found at '{}', please check.".format(args.resume))
            return

    # Clear these out
    '''
    import ipdb; ipdb.set_trace()
    from thop import profile
    dummy_input = [torch.randn(1,3,256,256).cuda()]
    flops, params = profile(model, inputs=dummy_input, verbose=True)
    '''

    if args.evaluate:
        # assert args.resume, "You should provide a checkpoint through args.resume."
        psnr, ssim, loss = infer(model, test_loader, criterion, args, logging)
        logging.info("Average PSNR {}, Average SSIM {}, Average Loss {}"
                     .format(psnr, ssim, loss))
        return

    # training progress
    for epoch in range(1 if not start_epoch else start_epoch, epochs+1):
        adjust_learning_rate(optimizer, epoch, lr_schedule)
        # train one epoch
        logging.info('Epoch [%d/%d]  lr: %e', epoch, epochs,
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
            best_psnr_epoch = epoch
            best_psnr = psnr
        if ssim > best_ssim:
            best_names.append('best_ssim.pth.tar')
            best_ssim_epoch = epoch
            best_ssim = ssim
        if loss < best_loss:
            best_names.append('best_loss.pth.tar')
            best_loss_epoch = epoch
            best_loss = loss
        if args.save_flag:
            utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'psnr': psnr,
            'ssim': ssim,
            'loss': loss,
            'optimizer': optimizer.state_dict()}, best_names, args.save_path)

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
        batch_size = inputs.size(0)
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            psnr = calc_PSNR(outputs, targets)
            ssim = calc_SSIM(torch.clamp(outputs,0,1), targets)

        # display
        Loss.update(loss.item(), batch_size)
        Batch_time.update(time.time() - end)
        PSNR.update(psnr, batch_size)
        SSIM.update(ssim, batch_size)
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
            batch_size = inputs.size(0)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            ssim = calc_SSIM(torch.clamp(outputs,0,1), targets)
            psnr = calc_PSNR(outputs, targets)
            PSNR.update(psnr, batch_size)
            SSIM.update(ssim, batch_size)
            Loss.update(loss.item(), batch_size)
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


if __name__ == '__main__':
    main()
