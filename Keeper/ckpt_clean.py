# 这个脚本用来去掉thop里烦人的total_ops & total_params等
import torch
import argparse

import models

parser = argparse.ArgumentParser("☆ Ease out the ANNOYING element! ☆")
parser.add_argument('-p', '--path', required=True)
parser.add_argument('-s', '--save_path', default='checkpoint.pth.tar')
parser.add_argument('-i', '--inplace', action='store_true')
args = parser.parse_args()

checkpoint = torch.load(args.path)

# remove additional elements introduced by thop
if 'total_ops' in checkpoint['state_dict'].keys():
    checkpoint['state_dict'].pop('total_ops')
    checkpoint['state_dict'].pop('total_params')

if args.save_path:
    torch.save(checkpoint, args.save_path)
