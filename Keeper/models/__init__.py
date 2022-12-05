import os
import importlib
import os.path as osp

from .edsr import *
from .unet import *
from .MPRNet_deblur import *
from .ResUNet import *
from .nafnet import *

this_folder = osp.dirname(osp.abspath(__file__))
# scan all the files under the model folder with '.py' in file names
model_filenames = [
    f for f in os.listdir(this_folder) if f.endswith('.py') and not f.startswith('__')
]
model_names = [f.split('.')[0] for f in model_filenames]


def get_module(model_type):
    # import all the model modules
    _model_modules = [
        importlib.import_module(f'models.{file_name}') for file_name in model_names
    ]
    candidate_path = []
    for module in _model_modules:
        if hasattr(module, model_type):
            candidate_path.append(module)

    if len(candidate_path) != 1:
        raise ValueError(f'Can\'t locate {model_type}, please check.')

    return candidate_path[0].__file__
