# 这个脚本用来去掉thop里烦人的total_ops & total_params等
import torch
import argparse

import models

parser = argparse.ArgumentParser("☆ Ease out the ANNOYING element! ☆")
parser.add_argument('-p', '--path', required=True)
parser.add_argument('-s', '--save_path', default='checkpoint.pth.tar')
args = parser.parse_args()

checkpoint = torch.load(args.path)

# remove additional elements introduced by thop
import ipdb; ipdb.set_trace()
print('OK')
