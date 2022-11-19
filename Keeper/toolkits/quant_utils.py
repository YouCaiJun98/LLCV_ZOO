from typing import Union

import torch

def get_qscheme(bit:int=8, symmetry:bool=True, per_channel:bool=True, pot_scale:bool=False, ch_axis:int=None):
    return {'bit': bit,
            'symmetry': symmetry,
            'per_channel': per_channel,
            'pot_scale': pot_scale,
            'ch_axis':ch_axis}

def get_qconfig_dict(name:str, w_observer:str='MinMaxObserver', a_observer:str='EMAMinMaxObserver',
                     w_fakequantize:str='LearnableFakeQuantize', w_fakeq_params:dict={},
                     a_fakequantize:str='LearnableFakeQuantize', a_fakeq_params:dict={},
                     w_qscheme:dict=get_qscheme(), a_qscheme:dict=get_qscheme(per_channel=False)):
    postfix = '_qconfig_dict'
    qconfig_name = name + postfix
    qconfig_dict = {'w_observer': w_observer,
                    'a_observer': a_observer,
                    'w_fakequantize': w_fakequantize,
                    'w_fakeq_params': w_fakeq_params,
                    'a_fakequantize': a_fakequantize,
                    'a_fakeq_params': a_fakeq_params,
                    'w_qscheme': w_qscheme,
                    'a_qscheme': a_qscheme}
    global prepare_custom_config_dict
    prepare_custom_config_dict.update({qconfig_name: qconfig_dict})

# Some extra setting dicts, currently the description is borrowed from mqbench.
"""
    prepare_custom_config_dict : {
    > extra_qconfig_dict: A dict that describe quantization configurations for a module 
        or the whole model.
        * Notice that we preserve some key words of `qconfig_dict` for special usage:
            - 'io' refers to the configuration of the first and last layer.
            - 'uniform','default' refer to the case that a module isn't 
                specified a certain configuration, this could be seen when 
                you need to uniformly quantize the whole model, or you specifically
                quantize some layers, while leaving the other layers quantize uniformly. 
        * A qconfig_dict should look like:
            >>>'extra_qconfig_dict' : {
                'w_observer': 'MinMaxObserver',
                'a_observer': 'EMAMinMaxObserver',
                'w_fakequantize': 'LearnableFakeQuantize',
                'w_fakeq_params': {},
                'a_fakequantize': 'LearnableFakeQuantize',
                'a_fakeq_params': {},
                'w_qscheme': {
                    'bit': 8,
                    'symmetry': True,
                    'per_channel': True,
                    'pot_scale': False},
                'a_qscheme': {
                    'bit': 8,
                    'symmetry': True,
                    'per_channel': False,
                    'pot_scale': False}
                },

    > extra_quantizer_dict: Extra params for quantizer.
    > reserve_attr: Dict, Specify attribute of model which should be preserved 
                after prepare. Since symbolic_trace only store attributes which is 
                in forward. If model.func1 and model.backbone.func2 should be preserved,
                {": ["func1"], "backbone": ["func2"] } should work."
    Attr below is inherited from Pytorch.
    > concrete_args: Specify input for model tracing.
    > extra_fuse_dict: Specify extra fusing patterns and functions.
    }
"""
prepare_custom_config_dict = {}

def calibrate(cali_loader, model, logger, args):
    model.eval()
    logger.info("Start calibration ...")
    logger.info("Calibrate images number = {}".format(len(cali_loader.dataset)))
    with torch.no_grad():
        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(cali_loader):
            if args.gpu is not None:
                noisy_imgs = noisy_imgs.cuda()
            restored_imgs = model(noisy_imgs)
            logger.info("Calibration ==> {}".format(batch_idx+1))
    logger.info("End calibration.")
    return


