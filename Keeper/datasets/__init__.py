import os
import math
import importlib
import numpy as np
from functools import partial

import torch

from .BSD   import *
from .SIDD  import *
from .SID   import *
from .GoPro import *

from toolkits import get_logger, get_dist_info

__all__ = ['get_dataloader']


# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
dataset_folder = os.path.dirname(os.path.abspath(__file__))
dataset_filenames = [
    f.split('.')[0] for f in os.listdir(dataset_folder) \
    if f.endswith('.py') and not f.startswith('__')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'datasets.{file_name}')
    for file_name in dataset_filenames
]



def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt['type']
    # dynamic instantiation
    dataset_cls = None
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(dataset_opt)

    logger = get_logger()
    logger.info(f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]}'
                 ' is created.')

    return dataset


def create_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()

    if phase == 'train':
        if dist:
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            persistent_workers=True,
        )
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed
        ) if seed is not None else None
    elif phase in ['val', 'test']:
        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

    else:
        raise ValueError(f'Wrong dataset phase: {phase}. '
                         "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: '
                    f'num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def get_dataloader(args):
    # There can be multiple test_loaders but only one train_loader and one val_loader.
    train_loader, val_loader, test_loader = None, {}, {}

    # TODO: this partition is wrong!
    if args.cfg['train_settings']['trainer_type'] == 'epoch':
        dataset_settings = args.cfg['train_settings']['datasets']
        batch_size = dataset_settings['train']['batch_size_per_gpu']
        num_workers = dataset_settings['train']['num_worker_per_gpu']
        loader_settings = {
            'num_workers': num_workers,
            'pin_memory': True,
            'batch_size': batch_size,
        }

        '''
        # SID-Sony
        img_list_files = ['./Sony/Sony_train_list.txt',
                          './Sony/Sony_val_list.txt',
                          './Sony/Sony_test_list.txt']
        train_data = SID_Sony(args.dataset, img_list_files[0], patch_size=args.patch_size,  data_aug=True,  stage_in='raw', stage_out='raw')
        val_data   = datasets.SID_Sony(args.dataset, img_list_files[1], patch_size=None, data_aug=False, stage_in='raw', stage_out='raw')
        test_data  = datasets.SID_Sony(args.dataset, img_list_files[2], patch_size=None, data_aug=False, stage_in='raw', stage_out='raw')
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=workers, pin_memory=True)
        '''

        # SIDD
        train_data = SIDD_sRGB_Train_DataLoader(dataset_settings['train']['root'], 96000, 256, True)
        val_data = SIDD_sRGB_Val_DataLoader(dataset_settings['val']['root'])
        test_data  = SIDD_sRGB_mat_Test_DataLoader(dataset_settings['test']['root'])
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **loader_settings)
        val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **loader_settings)
        test_loader  = torch.utils.data.DataLoader(test_data, shuffle=False, **loader_settings)

        '''
        # GoPro
        train_data = datasets.GoPro_sRGB_Train_DataSet(os.path.join(args.root, 'train'), 256)
        test_data = datasets.GoPro_sRGB_Test_DataSet(os.path.join(args.root, 'test'))
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **Loader_Settings)
        Loader_Settings['batch_size'] = 4
        val_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **Loader_Settings)
        test_loader  = torch.utils.data.DataLoader(test_data,  shuffle=False, **Loader_Settings)
        '''

        if args.multi_cards:
            # since the specified batch size is per GPU, there's no need to divide again.
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            train_loader = torch.utils.data.DataLoader(train_data, **loader_settings, sampler=train_sampler)

    else:
        # TODO: Here we fix iter training dataset as lmdb formatted.
        from .data_sampler import EnlargedSampler
        # from basicsr.data import create_dataset
        for phase, dataset_settings in args.cfg['datasets'].items():
            if 'train' in phase:
                # TODO: dataset enlarge ratio not implemented!
                dataset_enlarge_ratio = dataset_settings.get('dataset_enlarge_ratio', 1)
                dataset_settings['phase'] = 'train'
                train_data = create_dataset(dataset_settings)
                train_sampler = EnlargedSampler(train_data, args.world_size,
                                                args.local_rank, dataset_enlarge_ratio)
                train_loader = create_dataloader(
                    train_data,
                    dataset_settings,
                    num_gpu=args.world_size,
                    dist=args.multi_cards,
                    sampler=train_sampler,
                    seed=args.seed)

                args.num_iter_per_epoch = math.ceil(
                    len(train_data) * dataset_enlarge_ratio /
                    (dataset_settings['batch_size_per_gpu'] * args.world_size))
                args.total_iters = int(args.cfg['train_settings']['train_len'])
                args.total_epochs = math.ceil(args.total_iters / (args.num_iter_per_epoch))

                args.logging.info(
                    'Training statistics:'
                    f'\n\tNumber of train images: {len(train_data)}'
                    f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                    f'\n\tBatch size per gpu: {dataset_settings["batch_size_per_gpu"]}'
                    f'\n\tWorld size (gpu number): {args.world_size}'
                    f'\n\tRequire iter number per epoch: {args.num_iter_per_epoch}'
                    f'\n\tTotal epochs: {args.total_epochs}; iters: {args.total_iters}.')

            elif 'val' in phase:
                dataset_settings['phase'] = 'val'
                val_data = create_dataset(dataset_settings)
                val_loader.update({
                    dataset_settings['name']:create_dataloader(
                        val_data,
                        dataset_settings,
                        num_gpu=args.world_size,
                        dist=args.multi_cards,
                        sampler=None,
                        seed=args.seed)
                })
                args.logging.info(
                    'Validation statistics:'
                    f'Number of val images/folders in {dataset_settings["name"]}: '
                    f'{len(val_data)}'
                )

            elif 'test' in phase:
                dataset_settings['phase'] = 'test'
                test_data = create_dataset(dataset_settings)
                test_loader.update({
                    dataset_settings['name']:create_dataloader(
                        test_data,
                        dataset_settings,
                        num_gpu=args.world_size,
                        dist=args.multi_cards,
                        sampler=None,
                        seed=args.seed)
                })
                args.logging.info(
                    'Testing statistics:'
                    f'Number of val images/folders in {dataset_settings["name"]}: '
                    f'{len(test_data)}'
                )

            else:
                raise ValueError(f'Dataset phase {phase} is not recognized.')
    return train_loader, val_loader, test_loader
