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
    file_handler = logging.FileHandler(os.path.join(args.save_path, "test.log" if 
                                                    args.evaluate else "train.log"))
    #file_format = "'%(asctime)s %(levelname)s %(message)s'"
    file_handler.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
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

def calc_PSNR(inputs:torch.Tensor, targets:torch.Tensor,
              data_range:Union[int, float]=1.0) -> float:
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
    # Constant for numerical stability, could guarantee accuracy in .5f
    eps = 1e-10
    inputs = torch.clamp(inputs, 0, float(data_range))
    inputs, targets = inputs/float(data_range), targets/float(data_range)
    MSE = torch.mean((inputs - targets) ** 2, dim=[1, 2, 3])
    PSNR: torch.Tensor = - 10 * torch.log10(MSE + eps)
    return PSNR.mean(dim=0).item()

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

def save_checkpoint(state:dir, best_names:str, save_dir:str, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    if best_names:
        for best_name in best_names:
            best_path = os.path.join(save_dir, best_name)
            shutil.copyfile(save_path, best_path)

