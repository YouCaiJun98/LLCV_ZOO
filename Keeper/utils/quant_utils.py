def get_qscheme(bit:int=8, symmetry:bool=True, per_channel:bool=True, pot_scale:bool=False):
    return {'bit': bit,
            'symmetry': symmetry,
            'per_channel': per_channel,
            'pot_scale': pot_scale}

def get_qconfig_dict(prepare_custom_config_dict, name:str, w_observer:str='MinMaxObserver', a_observer:str='EMAMinMaxObserver',
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
    prepare_custom_config_dict.update({qconfig_name: qconfig_dict})
    return prepare_custom_config_dict

