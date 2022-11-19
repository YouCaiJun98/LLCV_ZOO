# Utils to estimate peak memory of a given model.
# could be applied to U-Net, should extend to more types of models.
import tqdm
import numpy as np

import torch

# we use normal unet for different pruning methods.
from .unet_dependency import *

# dicts below should be given in advance.
global module_name
global downsampled_layer
global module_dependency

global scale_1
global scale_1_2
global scale_1_4
global scale_1_8
global scale_1_16

global init_channels

def peak_mem(model, patch_size, map_acquire=False, unit=None):
    peak_mem = 0
    peak_mem_module = None
    module_dict = {}
    # should be adjust to the corresponding right name
    init_channels = model.conv1_1.in_channels

    # update module_dict, calculate output feature memory of every module.
    for name in module_name:
        layer = getattr(model, name)
        module_dict.update({name: {}})
        if name in scale_1:
            scale = 1
        elif name in scale_1_2:
            scale = 2
        elif name in scale_1_4:
            scale = 4
        elif name in scale_1_8:
            scale = 8
        elif name in scale_1_16:
            scale = 16
        # batch size is fixed as 1.
        if hasattr(layer, 'pwc'):
            layer.out_channels = layer.pwc.out_channels
        out_feat_mem = layer.out_channels * (patch_size / scale) ** 2
        module_dict[name].update({'out_channels': layer.out_channels})
        module_dict[name].update({'out_feat_mem': out_feat_mem})
        # in case that the skipped feature is not of the same size of that passed through
        # the main branch.
        if name in downsampled_layer:
            module_dict.update({name + '_downsampled': {}})
            module_dict[name + '_downsampled'].update({'out_feat_mem': out_feat_mem / 4})

    # calculate mem overhead of every layer
    for name in module_name:
        total_mem = 0
        for dependent_name in module_dependency[name]:
            if dependent_name == 'input':
                total_mem += init_channels * patch_size ** 2
            else:
                total_mem += module_dict[dependent_name]['out_feat_mem']
        module_dict[name].update({'total_mem': total_mem})

    # unit transfer
    if unit is not None:
        assert unit in ['MB'], 'Currently we only support specify MB as unit.'

        if unit == 'MB':
            for name in module_name:
                mem = module_dict[name]['total_mem'] * 4 / 1024 / 1024
                module_dict[name]['total_mem'] = mem

    # pick layer with the highest mem overhead
    for name in module_name:
        if module_dict[name]['total_mem'] > peak_mem:
            peak_mem = module_dict[name]['total_mem']
            peak_mem_module = name
    if map_acquire:
        return module_dict
    return peak_mem_module, peak_mem


def mem_map(model, patch_size, unit='MB'):
    mem_map = {}
    raw_map = peak_mem(model, patch_size, True)
    for name in module_name:
        if unit == 'MB':
            mem = raw_map[name]['total_mem'] * 4 / 1024 / 1024
        mem_map.update({name: mem})
    return mem_map


def calc_batch_size(model, peak_mem, mem_budget):
    params = model.total_params
    params_mem = params * 4 / 1024 / 1024 # MB
    feat_mem_budget = mem_budget - params_mem
    batch_size = np.ceil(feat_mem_budget / peak_mem)
    return batch_size

def eval_latency(model, batch_size, patch_size = 256):
    repetitions = 20
    random_input = torch.randn((int(batch_size), 3, patch_size, patch_size)).cuda()
    # warm up gpu
    with torch.no_grad():
        for _ in range(5):
            _ = model(random_input)
    # synchronize testing
    torch.cuda.synchronize()

    # from https://zhuanlan.zhihu.com/p/460524015
    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = model(random_input)
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time
    avg = timings.sum() / repetitions
    return avg
