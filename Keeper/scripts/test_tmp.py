#!/usr/bin/env python
# 我猜这个应该是当时DIRNet的测试脚本
import argparse
import os
import sys
import time
import scipy.io
import torch
import torch.nn as nn

import datasets
import models.MPRNet as LSID
# import models.lsid_stridedconv_1 as LSID
from trainer_denoise import Trainer, Validator
from datasets.SIDD import *
import utils
import tqdm

configurations = {
    1: dict(
        max_iteration=240000,
        lr=1e-4,
        momentum=0.9,
        weight_decay=1e-8,
        gamma=0.5,
        step_size=60000, # "lr_policy: step"
        interval_validate=3000,
    ),
}

def get_parameters(model, bias=False):
    for k, m in model._modules.items():
        print("get_parameters", k, type(m), type(m).__name__, bias)
        if bias:
            if isinstance(m, nn.Conv2d):
                yield m.bias
        else:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                yield m.weight

def main():
    parser = argparse.ArgumentParser("Learning to See in the Dark PyTorch")
    parser.add_argument('cmd', type=str,  choices=['train', 'test'], help='train or test')
    parser.add_argument('--arch_type', type=str, default='Sony', help='camera model type', choices=['Sony', 'Fuji'])
    parser.add_argument('--dataset_dir', type=str, default='./datasets/SIDD/SIDD_patches', help='dataset directory')
    parser.add_argument('--log_file', type=str, default='./checkpoint/denoise/test.log', help='log file')
    parser.add_argument('--gt_png', action='store_true', help='uses preconverted png file as ground truth')
    parser.add_argument('--use_camera_wb', action='store_true', help='converts train RAW file to png')
    parser.add_argument('--valid_use_camera_wb', action='store_true', help='converts valid RAW file to png')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/denoise/',
                        help='checkpoints directory')
    parser.add_argument('--result_dir', type=str, default='./checkpoint/denoise/',
                        help='directory where results are saved')
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                        help='the number of settings and hyperparameters used in training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='batch size in validation')
    parser.add_argument('--test_batch_size', type=int, default=32, help='batch size in test')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size')
    parser.add_argument('--save_freq', type=int, default=10, help='checkpoint save frequency')
    parser.add_argument('--print_freq', type=int, default=10, help='log print frequency')
    parser.add_argument('--upper_train', type=int, default=-1, help='max of train images(for debug)')
    parser.add_argument('--upper_valid', type=int, default=-1, help='max of valid images(for debug)')
    parser.add_argument('--upper_test', type=int, default=-1, help='max of test images(for debug)')
    parser.add_argument('--resume', type=str, default='', help='checkpoint file(for training or test)')
    parser.add_argument('--tf_weight_file', type=str, default='', help='weight file ported from TensorFlow')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--pixel_shuffle', action='store_true',
                        help='uses pixel_shuffle in training')
    args = parser.parse_args()

    if args.cmd == 'train':
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        cfg = configurations[args.config]
    if args.cmd == 'test':
        # specify one of them
        assert args.tf_weight_file or args.resume
        assert not(args.tf_weight_file and args.resume)

    log_file = args.log_file
    resume = args.resume
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    if cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    root = args.dataset_dir

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    dataset_class = datasets.__dict__[args.arch_type]
    if args.cmd == 'train':
        dt = SIDD_Medium_sRGB_Train_DataLoader(os.path.join(args.dataset_dir, 'train'), 96000, 256, True)
        train_loader = torch.utils.data.DataLoader(dt, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        dv = SIDD_sRGB_Val_DataLoader(os.path.join(args.dataset_dir, 'val'))
        val_loader = torch.utils.data.DataLoader(dv, batch_size=args.valid_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.cmd == 'test':
        dt = SIDD_sRGB_mat_Test_DataLoader(os.path.join(args.dataset_dir, 'test'))
        test_loader = torch.utils.data.DataLoader(dt, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 2. model
    if 'Fuji' in args.arch_type:
        model = LSID.lsid(inchannel=9, block_size=3)
    else: # Sony
        model = LSID.lsid(inchannel=3, block_size=1)
    print(model)

    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        checkpoint['arch'] = args.arch_type
        assert checkpoint['arch'] == args.arch_type
        print("Resume from epoch: {}, iteration: {}".format(start_epoch, start_iteration))
    else:
        if args.cmd == 'test':
            utils.load_state_dict(model, args.tf_weight_file) # load weight values

    if cuda:
        model = model.cuda()

    criterion = nn.L1Loss()
    if cuda:
        criterion = criterion.cuda()

    # 3. optimizer
    if args.cmd == 'train':
        optim = torch.optim.Adam(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True), 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            ],
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay'])
        if resume:
            optim.load_state_dict(checkpoint['optim_state_dict'])
    
        # lr_policy: step
        last_epoch = start_iteration if resume else -1
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,  cfg['step_size'],
                                                       gamma=cfg['gamma'], last_epoch=last_epoch)

    if args.cmd == 'train':
        trainer = Trainer(
            cmd=args.cmd,
            cuda=cuda,
            model=model,
            criterion=criterion,
            optimizer=optim,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            log_file=log_file,
            max_iter=cfg['max_iteration'],
            checkpoint_dir=args.checkpoint_dir,
            result_dir=args.result_dir,
            use_camera_wb=args.use_camera_wb,
            print_freq=args.print_freq,
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()
    elif args.cmd == 'test':
        validator = Validator(
            cmd=args.cmd,
            cuda=cuda,
            model=model,
            criterion=criterion,
            val_loader=test_loader,
            log_file=log_file,
            result_dir=args.result_dir,
            use_camera_wb=args.use_camera_wb,
            print_freq=args.print_freq,
        )
        validator.validate()

if __name__ == '__main__':
    main()
