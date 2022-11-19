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
from torch.nn.parallel import DistributedDataParallel as DDP

import toolkits
import models
import datasets

### -------------------- Parser Zone  -------------------- ###
parser = argparse.ArgumentParser("☆ Welcome to the ZOO of LLCV ☆")
parser.add_argument('-a', '--arch', metavar='ARCH', default='unet')
parser.add_argument('--root', type=str, default='./datasets/SIDD/SIDD_patches', help='root location of the data corpus')
parser.add_argument('--save_path', type=str, default='./checkpoints', help='parent path for saved experiments')
parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: None)')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint path of previous model, loading for evaluation or retrain')
parser.add_argument('--eager_test', type=int, default=None, dest='eager_test',
                    help='debug only. test per 10 epochs during training')
parser.add_argument('-c', '--configuration', required=True, help='model & train/validate settings')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-d', '--debug', dest='save_flag', action='store_false',
                    help='if specified, logs won\'t be saved')
parser.add_argument('--local_rank', default=0, type=int,
                    help='Local rank for each gpu under distribution environment.')

def main():
    # get model dicts
    model_names = sorted(name for name in models.__dict__ \
                if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

    # parse the instructions
    args = parser.parse_args()
    # To distinguish master and slave under distributed training scene.
    args.slave = True if args.local_rank != 0 else False

    # parse configurations
    with open(args.configuration, 'r') as rf:
        cfg = yaml.load(rf, Loader=yaml.FullLoader)
        args.cfg = cfg
        train_cfg = cfg['train_settings']
    gpu = str(train_cfg['gpu'])
    seed = train_cfg['seed'] if train_cfg['seed'] else None

    if args.save_flag:
        save_name = cfg['exp_name']
        # create dir for saving results
        args.save_name = '{}-{}'.format(save_name, 'test' if args.evaluate else 'train')
        if cfg.get('timestamp_enabled', True):
            args.save_name += '-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
        args.save_path = os.path.join(args.save_path, args.save_name)
        # scripts & configurations to be saved
        save_list = ['models/unet_mosp.py'] + [__file__] + [args.configuration]
        if args.local_rank == 0:
            toolkits.create_exp_dir(args.save_path, scripts_to_save=save_list)
    import ipdb; ipdb.set_trace()
    # get info logger
    logging = toolkits.get_logger(args)
    args.logging = logging


    # set up device.
    if gpu:
        args.gpu_flag = True
        args.multi_cards = False if len(gpu) == 1 else True
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        if len(gpu) == 1:
            args.world_size = 1
            device = torch.device('cuda')
            torch.cuda.set_device(0) # single card case.
        elif len(gpu) > 1:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl')
            device = torch.device('cuda', args.local_rank)
            args.world_size = torch.distributed.get_world_size() # batch size in cfg file refers to the total size.
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
        args.seed = seed
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

    # set up the trainer
    trainer = toolkits.create_trainer(args)
    import ipdb; ipdb.set_trace()

    from thop import profile
    rinput = torch.randn(1,3,256,256).cuda()
    flops, params = profile(trainer.model, (rinput,))
    from toolkits.mem_utils import peak_mem
    _, pmem = peak_mem(trainer.model, 256, unit='MB')

    if args.evaluate:
        assert args.resume, "You should provide a checkpoint through args.resume."
        psnr, ssim, loss = trainer.val()
        logging.info("Average PSNR {}, Average SSIM {}, Average Loss {}"
                     .format(psnr, ssim, loss))
        return

    # training progress
    else:
        trainer.train()


if __name__ == '__main__':
    main()
