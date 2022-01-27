from typing import Union
import logging
import os
import sys
import shutil
import torch

def get_logger(args):
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger()
    if args.save_flag:
        file_handler = logging.FileHandler(os.path.join(args.save_path, "test.log" if 
                                                        args.evaluate else "train.log"))
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    logger.info("Conducting Command: %s", " ".join(sys.argv))
    return logger

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
             dst_file = os.path.join(path, 'scripts', os.path.basename(script))
             shutil.copyfile(script, dst_file)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger
        assert self.logger, 'You should provide a logger for display.'

    def display(self, batch, batch_name=None):
        entries = [self.prefix + batch_name + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state:dir, best_names:str, save_dir:str, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    if best_names:
        for best_name in best_names:
            best_path = os.path.join(save_dir, best_name)
            shutil.copyfile(save_path, best_path)

from collections import OrderedDict
def get_state_dict(checkpoint):
    if type(checkpoint) is OrderedDict:
        state_dict = checkpoint
    elif type(checkpoint) is dict:
        for key in checkpoint.keys():
            # In case the params of optimizer is saved in the checkpoint
            if ('state_dict' in key) and ('opti' not in key):
                state_dict = checkpoint[key]
    return state_dict


