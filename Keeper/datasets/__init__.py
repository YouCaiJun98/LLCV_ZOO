import os
import math

import torch

from .BSD   import *
from .SIDD  import *
from .utils import *
from .SID   import *
from .GoPro import *

def get_dataloader(args):
    train_loader, val_loader, test_loader = None, None, None

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
        from basicsr.data import create_dataloader, create_dataset
        for phase, dataset_settings in args.cfg['train_settings']['datasets'].items():
            if phase == 'train':
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

            elif phase == 'val':
                dataset_settings['phase'] = 'val'
                val_data = create_dataset(dataset_settings)
                val_loader = create_dataloader(
                    val_data,
                    dataset_settings,
                    num_gpu=args.world_size,
                    dist=args.multi_cards,
                    sampler=None,
                    seed=args.seed)
                args.logging.info(
                    'Validation statistics:'
                    f'Number of val images/folders in {dataset_settings["name"]}: '
                    f'{len(val_data)}'
                )

            elif phase == 'test':
                dataset_settings['phase'] = 'test'
                test_data = create_dataset(dataset_settings)
                test_loader = create_dataloader(
                    test_data,
                    dataset_settings,
                    num_gpu=args.world_size,
                    dist=args.multi_cards,
                    sampler=None,
                    seed=args.seed)
                args.logging.info(
                    'Testing statistics:'
                    f'Number of val images/folders in {dataset_settings["name"]}: '
                    f'{len(test_data)}'
                )

            else:
                raise ValueError(f'Dataset phase {phase} is not recognized.')
    return train_loader, val_loader, test_loader
