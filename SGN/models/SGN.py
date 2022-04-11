import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SGN', 'sgn_l1', 'sgn_l2', 'sgn_l3', 'sgn_l4']

class SGN(nn.Module):
    def __init__(self, hyper_params):
        super(SGN, self).__init__()
        self.g = hyper_params['g']
        self.m = hyper_params['m']
        self.n = hyper_params['n']
        self.scale = hyper_params['scale']


def sgn_l1():
    return

def sgn_l2():
    return

def sgn_l3():
    return

def sgn_l4():
    return
