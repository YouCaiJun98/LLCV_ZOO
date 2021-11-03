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
from datasets.BSD import bsd200, bsd100, bsd68
from SSIM import ssim as calc_SSIM
from models.dynamic_unet import d_unet, ActivationRateAccum

parser = argparse.ArgumentParser("Dynamic Resolution Network for Denoising.")
parser.add_argument('--dataset', type=str, default='./datasets/BSD500/', help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')
parser.add_argument('--steps', type=int, default=100, help='steps of each epoch')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--save_name', type=str, default='EXP', help='experiment name')
parser.add_argument('--save_path', type=str, default='./checkpoints', help='parent path for saved experiments')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--patch_size', type=int, default=64, help='patch size')
parser.add_argument('--gpu', type=str, default='0', help='gpu device ids')
parser.add_argument('--seed', type=int, default=99, help='seed for initializing training')
parser.add_argument('--sigma', type=int, default=25, help='noise level')
parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: None)')
parser.add_argument('--resume', type=str, default=None, 
                    help='checkpoint path of previous model, loading for evaluation or retrain')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args()

# currently we fix pass rate for 4 gate units as 0.9, 0.8, 0.7, 0.6
target_rates = [0.9, 0.8, 0.7, 0.6]
# control trade-off between act & performance, currently fixed as 1
act_factor = 1


def main():
    # create dir for saving results
    args.save_name = '{}-{}-{}'.format('test' if args.evaluate else 'train',
                                        args.save_name, time.strftime("%Y%m%d-%H%M%S"))
    args.save_path = os.path.join(args.save_path, args.save_name)
    utils.create_exp_dir(args.save_path, scripts_to_save=glob.glob('*.py'))

    # get info logger
    logging = utils.get_logger(args)

    # set up device.
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        gpus = [int(d) for d in args.gpu.split(',')]
        torch.cuda.set_device(gpus[0]) # currently only support one card run
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
    
    # set up model/optimizer/dataset
    model = d_unet(inchannel=1)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    criterion = nn.L1Loss().cuda()
    #criterion = nn.MSELoss().cuda()

    train_data = bsd200(gray=True, pth=args.dataset+'train/', length=args.steps*args.batch_size,
                        patch_size=args.patch_size, sigma=args.sigma, rnd_aug=True)
    val_data = bsd100(gray=True, pth=args.dataset+'val/', length=50*args.batch_size,
                        patch_size=args.patch_size, sigma=args.sigma, rnd_aug=True)
    test_data = bsd200(gray=True, pth=args.dataset+'test/', length=200, sigma=args.sigma,
                       patch_size=args.patch_size, rnd_aug=True if args.patch_size else False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, pin_memory=True)

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
            if not args.gpu:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch=best_psnr_epoch=best_ssim_epoch=best_loss_epoch = checkpoint['epoch']
            best_psnr, best_ssim, best_loss = checkpoint['psnr'], checkpoint['ssim'], checkpoint['loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), 
                                                                   last_epoch=start_epoch)
            logging.info("Loaded checkpoint '{}' (epoch{}), PSNR {} dB, SSIM {}, Loss {}."
                        .format(args.resume, checkpoint['epoch'], checkpoint['psnr'], checkpoint['ssim'],
                                checkpoint['loss']))
        else:
            logging.info("No checkpoint found at '{}', please check.".format(args.resume))
            return

    if args.evaluate:
        assert args.resume, "You should provide a checkpoint through args.resume."
        # import ipdb;ipdb.set_trace()
        psnr, ssim, loss = infer(model, test_loader, criterion, args, logging)
        logging.info("Average PSNR {}, Average SSIM {}, Average Loss {}"
                     .format(psnr, ssim, loss))
        return
 
    # training progress
    for epoch in range(0 if not start_epoch else start_epoch, args.epochs):        
        # train one epoch
        logging.info('Epoch [%d/%d]  lr: %e', epoch+1, args.epochs, scheduler.get_last_lr()[0])
        logging.info('<-Training Phase->')
        train(model, train_loader, criterion, optimizer, args, logging)

        # validate last epoch
        logging.info('<-Validating Phase->')
        psnr, ssim, loss = infer(model, val_loader, criterion, args, logging)

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

        scheduler.step()
        logging.info('PSNR:%4f SSIM:%4f Loss:%4f / Best_PSNR:%4f Best_SSIM:%4f Best_Loss:%4f', 
                     psnr, ssim, loss, best_psnr, best_ssim, best_loss)
    logging.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)', 
                 best_loss, best_loss_epoch, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch)

def train(model, train_loader, criterion, optimizer, args, logging):
    Loss = utils.AverageMeter('total loss')
    Loss_a = utils.AverageMeter('activation loss')
    Loss_p = utils.AverageMeter('pixel loss')
    Activations = utils.AverageMeter('activaiton rates')
    Batch_time = utils.AverageMeter('batch time')

    # model state
    model.train()

    global target_rates
    global act_factor

    # timer
    end = time.time()

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs, activation_rates = model(inputs)

        # activation rate calculation
        act_loss = 0
        acts_list = []
        acts_ave = 0
        for i, act in enumerate(activation_rates):
            if target_rates[i] < 1:
                act_batchmean = torch.mean(act)
                acts_ave += act_batchmean
                acts_list.append(act_batchmean)
                act_loss += torch.pow(target_rates[i]-act_batchmean, 2)
            else:
                acts_ave += 1
                acts_list.append(1)

        acts_ave = torch.mean(acts_ave / len(activation_rates))
        act_loss = torch.mean(act_loss / len(activation_rates))
        pixel_loss = criterion(outputs, targets)
        loss = pixel_loss + act_factor * act_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # display
        Loss.update(loss.item(), inputs.size(0))
        Loss_a.update(act_loss.item(), inputs.size(0))
        Loss_p.update(pixel_loss.item(), inputs.size(0))
        Activations.update(acts_ave.item(), 1)
        Batch_time.update(time.time() - end)
        end = time.time()

        if args.print_freq is not None and step % args.print_freq == 0:
            logging.info('batch [{0}/{1}] \t'
                         'Time {Batch_time.val:.3f} ({Batch_time.avg:.3f})\t'
                         'Total/Pixel/ActRate Loss ({Loss.avg:.4f}/{Lossp.avg:.4f}/' \
                         '{Lossa.avg:.4f})  '
                         'Activations: {Activations.val:.3f} ({Activations.avg:.3f})  '
                         'Activation rate for each gate: {gate_rate}'
                         .format(step, len(train_loader), Batch_time=Batch_time, Loss=Loss,
                                 Lossp=Loss_p, Lossa=Loss_a, Activations=Activations,
                                 gate_rate=[i.item() for i in acts_list]))

def infer(model, val_loader, criterion, args, logging):
    Loss = utils.AverageMeter('total loss')
    Loss_a = utils.AverageMeter('activation loss')
    Loss_p = utils.AverageMeter('pixel loss')
    Activations = utils.AverageMeter('activaiton rates')
    Batch_time = utils.AverageMeter('batch time')
    PSNR = utils.AverageMeter('PSNR')
    SSIM = utils.AverageMeter('SSIM')
    accumulator = ActivationRateAccum(4)
    
    # timer
    end = time.time()

    # default settings for activation rates
    global target_rates
    global act_factor
    
    model.eval()
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, activation_rates = model(inputs)

            # calculate activation loss and total loss
            acts_ave = 0
            act_loss = 0
            for i, act in enumerate(activation_rates):
                if target_rates[i] < 1:
                    act_batchmean = torch.mean(act)
                    acts_ave += act_batchmean
                    act_loss += torch.pow(target_rates[i]-act_batchmean, 2)
                else:
                    acts_ave += 1

            accumulator.accumulate(activation_rates)
            acts_ave = torch.mean(acts_ave / len(activation_rates))
            act_loss = torch.mean(act_loss / len(activation_rates))
            pixel_loss = criterion(outputs, targets)
            loss = pixel_loss + act_factor * act_loss
            ssim = calc_SSIM(torch.clamp(outputs,0,1), targets)
            psnr = utils.calc_PSNR(outputs, targets)
            n = inputs.size(0)
            PSNR.update(psnr, n)
            SSIM.update(ssim, n)
            Loss_p.update(pixel_loss.item(), n)
            Loss_a.update(act_loss.item(), n)
            Loss.update(loss.item(), n)
            Activations.update(acts_ave.item(), 1)
            Batch_time.update(time.time() - end)
            end = time.time()

            if args.print_freq is not None and batch % args.print_freq == 0:
                logging.info('batch [{0}/{1}]  \t'
                             'Time {Batch_time.val:.3f} ({Batch_time.avg:.3f})\t'
                             'Total/Pixel/ActRate Loss ({Loss.avg:.4f}/{Lossp.avg:.4f}/' \
                             '{Lossa.avg:.4f})  PSNR {PSNR.val:.4f} ({PSNR.avg:.4f})\t'
                             'SSIM {SSIM.val:.4f} ({SSIM.avg:.4f})  '
                             'Activations: {Activations.val:.3f} ({Activations.avg:.3f})'
                             .format(batch, len(val_loader), Batch_time=Batch_time, Loss=Loss,
                                     Lossp=Loss_p, Lossa=Loss_a, PSNR=PSNR, SSIM=SSIM, 
                                     Activations=Activations))

    logging.info("Average activation rate for each gate unit is: {}".format(accumulator.getoutput()))

    return PSNR.avg, SSIM.avg, Loss.avg

if __name__ == '__main__':
    main()
