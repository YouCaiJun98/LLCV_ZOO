import os
import time
import math

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import models
import datasets
import toolkits
import toolkits.losses
from toolkits.metric import calc_SSIM, calc_PSNR

__all__ = ['create_trainer']

def create_trainer(args):
    """Create trainer.

    Args:
        args (dict): Configuration. It constains:
            trainer_type (str): Trainer type.
    """
    trainer_type = args.cfg['train_settings'].get('trainer_type')

    assert trainer_type in ['epoch', 'iter'], \
        f"The specified trainer_type [{trainer_type}] not in ['epoch', 'iter']!"

    if trainer_type == 'epoch':
        return EpochTrainer(args)

    elif trainer_type == 'iter':
        return IterTrainer(args)



class BaseTrainer():
    trainer_type = 'base'

    def __init__(self, args):
        self.args = args
        self.cfg  = args.cfg
        self.args.resume = self.cfg['train_settings'].get('resume', None)
        self.slave = self.args.slave
        self.device = self.args.device
        self.logging = self.args.logging
        self.disp_freq = self.cfg['display'].get('print_freq', 10)
        self.reset_counter()

        self.init_training_settings()

    def reset_counter(self):
        self.start_iter = 0
        self.start_epoch = 0
        self.best_psnr = 0
        self.best_psnr_epoch = 0
        self.best_ssim = 0
        self.best_ssim_epoch = 0
        self.best_loss = float('inf')
        self.best_loss_epoch = 0


    def init_training_settings(self):
        # Create model and (optionally) load the ckpt.
        self.create_model()
        # Resume a model if provided, only master should load ckpt.
        if self.args.resume is not None:
            self.load_checkpoint()

        # Set up optimizer, lr scheduler and criterion.
        self.create_optimizer()
        self.create_scheduler()
        self.create_criterion()

        # Get DataLoader
        self.get_dataloader()


    def create_model(self):
        model_names = sorted(name for name in models.__dict__ \
                if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
        model_type = self.cfg['model_settings'].pop('type')
        assert model_type in model_names, \
            f'The specified model [{model_type}] is not supported yet!'
        self.model = models.__dict__[model_type](self.cfg['model_settings'])
        self.logging.info(self.model)

        if self.args.gpu_flag:
            self.model.to(self.device)
            if self.args.multi_cards:
                self.model = DDP(self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank)
        else:
            self.logging.info("Using CPU. This will be slow.")


    def load_checkpoint(self):
        if self.args.resume and not self.slave:
            if os.path.isfile(self.args.resume):
                self.logging.info("Loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume, map_location=self.device)
                if 'state_dict' in checkpoint:
                    # clean the checkpoint 
                    if 'total_params' in checkpoint['state_dict']:
                        checkpoint['state_dict'].pop('total_params')
                        checkpoint['state_dict'].pop('total_ops')
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch']
                if 'iter' in checkpoint:
                    self.start_iter = checkpoint['iter']
        else:
            raise FileNotFoundError(
                f"No checkpoint found at '{self.args.resume}', please check.")


    def create_optimizer(self):
        # TODO: currently we only support single optimizer training.
        train_settings = self.cfg['train_settings']

        optim_params = []
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        self.optim_type = train_settings['optim'].pop('type')
        # TODO: maybe something more fancy.
        if self.optim_type == 'Adam':
            self.optimizer = torch.optim.Adam([{'params': optim_params}],
                                              **train_settings['optim'])

        elif self.optim_type == 'SGD':
            self.optimizer = torch.optim.SGD([{'params': optim_params}],
                                              **train_settings['optim'])

        elif self.optim_type == 'AdamW':
            self.optimizer = torch.optim.AdamW([{'params': optim_params}],
                                              **train_settings['optim'])

        else:
            raise NotImplementedError(
                f'optimizer {self.optim_type} is not supported yet.')


    def create_scheduler(self):
        # TODO: currently we only support single scheduler training.
        train_settings = self.cfg['train_settings']
        self.scheduler_type = train_settings['scheduler'].pop('type')
        if self.scheduler_type == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **train_settings['scheduler'])

        elif self.scheduler_type == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **train_settings['scheduler'])

        # TODO: wrap such flexible MultiStage LR as lr_scheduler.
        elif self.scheduler_type == 'FlexMultiStageLR':
            from toolkits.optimizer import adjust_learning_rate
            self.scheduler = adjust_learning_rate

        else:
            raise NotImplementedError(
                f'Scheduler {self.scheduler_type} is not implemented yet.')


    def get_dataloader(self):
        self.train_loader, self.val_loader, self.test_loader = \
            datasets.get_dataloader(self.args)


    def create_criterion(self):
        # TODO: currently we only support single loss training
        train_settings = self.cfg['train_settings']

        self.loss_type = train_settings['loss'].pop('type')
        if self.loss_type == 'L1Loss':
            self.criterion = nn.L1Loss()

        elif self.loss_type == 'PSNRLoss':
            self.criterion = toolkits.losses.PSNRLoss(**train_settings['loss'])

        else:
            raise NotImplementedError(
                f'Loss type {self.loss_type} is not implemented yet.')

        if self.args.gpu_flag:
            self.criterion.to(self.device)


    def train(self):
        pass

    def val(self):
        pass

    def get_val_status(self, test_flag):
        # During validation, we don't crop border or test on the y channel. 
        crop_border = 0
        test_y_channel = False
        if not test_flag:
            return crop_border, test_y_channel
        else:
            # default test case.
            if self.cfg.get('val_settings') is None:
                crop_border = 0
                test_y_channel = False
            else:
                val_cfg = self.cfg.get('val_settings')
                crop_border = val_cfg.get('crop_border', 0)
                test_y_channel = val_cfg.get('test_y_channel', False)
            return crop_border, test_y_channel

    def update_learning_rate(self, curr_iter=None, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration or epoch.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                                Default： -1.
        """
        # TODO: update scheduler type
        if self.scheduler_type == 'FlexMultiStageLR':
            lr_schedule = self.cfg['train_settings']['scheduler']['lr_schedule']
            self.scheduler(self.optimizer, curr_iter, lr_schedule)

        else:
            if curr_iter > 1:
                self.scheduler.step()

            # TODO: finish this
            # set up warm-up learning rate
            if curr_iter < warmup_iter:
                raise NotImplementedError(
                f'lr warmup is not implemented!')


class EpochTrainer(BaseTrainer):
    trainer_type = 'epoch'

    def __init__(self, args):
        super(EpochTrainer, self).__init__(args)
        self.epochs = self.cfg['train_settings']['train_len']

    def train_epoch(self):
        Loss = toolkits.AverageMeter('Loss')
        PSNR = toolkits.AverageMeter('PSNR')
        SSIM = toolkits.AverageMeter('SSIM')
        Batch_time = toolkits.AverageMeter('batch time')

        # model state
        self.model.train()

        # timer
        end = time.time()

        for step, (inputs, targets) in enumerate(self.train_loader):
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            with torch.no_grad():
                psnr = calc_PSNR(outputs, targets)
                ssim = calc_SSIM(torch.clamp(outputs,0,1), targets)

            # display
            Loss.update(loss.item(), batch_size)
            Batch_time.update(time.time() - end)
            PSNR.update(psnr, batch_size)
            SSIM.update(ssim, batch_size)
            end = time.time()

            if self.disp_freq is not None and step % self.disp_freq == 0:
                self.logging.info('batch [{0}/{1}] \t'
                         'Time {Batch_time.val:.3f} ({Batch_time.avg:.3f})\t'
                         'Loss {Loss.val:.3f} ({Loss.avg:.3f})\t'
                         'PSNR {PSNR.val:.3f} ({PSNR.avg:.3f})\t'
                         'SSIM {SSIM.val:.3f} ({SSIM.avg:.3f})'
                         .format(step, len(self.train_loader), Batch_time=Batch_time, Loss=Loss,
                         PSNR=PSNR, SSIM=SSIM))


    def train(self):
        for epoch in range(1 if not self.start_epoch else self.start_epoch, self.epochs+1):
            self.update_learning_rate(epoch)

            if self.args.gpu_flag and self.args.multi_cards:
                self.train_loader.sampler.set_epoch(epoch)

            # train one epoch
            self.logging.info('Epoch [%d/%d]  lr: %e', epoch, self.epochs,
                     self.optimizer.state_dict()['param_groups'][0]['lr'])

            self.logging.info('<-Training Phase->')

            # train one epoch
            self.train_epoch()

            # validate last epoch
            self.logging.info('<-Validating Phase->')
            if not self.slave:
                psnr, ssim, loss = self.val()

            # test one epoch if it's under eager mode
            if self.args.eager_test and epoch % self.args.eager_test == 0:
                self.logging.info('<-Testing Phase->')
                psnr_test, ssim_test, loss_test = self.val(True)
                for dataset_name in psnr_test.keys():
                    self.logging.info(f'Test - {dataset_name:}')
                    self.logging.info('PSNR:%4f SSIM:%4f Loss:%4f',
                                      psnr_test[dataset_name],
                                      ssim_test[dataset_name],
                                      loss_test[dataset_name])

            if not self.slave:
                # model save
                best_names = []
                if psnr > self.best_psnr and not math.isinf(psnr):
                    best_names.append('best_psnr.pth.tar')
                    self.best_psnr_epoch = epoch
                    self.best_psnr = psnr
                if ssim > self.best_ssim:
                    best_names.append('best_ssim.pth.tar')
                    self.best_ssim_epoch = epoch
                    self.best_ssim = ssim
                if loss < self.best_loss:
                    best_names.append('best_loss.pth.tar')
                    self.best_loss_epoch = epoch
                    self.best_loss = loss
                if self.args.save_flag:
                    toolkits.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict() if not self.args.multi_cards else self.model.module.state_dict(),
                    'psnr': psnr,
                    'ssim': ssim,
                    'loss': loss,
                    'optimizer': self.optimizer.state_dict()}, best_names, self.args.save_path)

                self.logging.info('PSNR:%4f SSIM:%4f Loss:%4f / Best_PSNR:%4f Best_SSIM:%4f Best_Loss:%4f',
                        psnr, ssim, loss, self.best_psnr, self.best_ssim, self.best_loss)
        self.logging.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)',
                 self.best_loss, self.best_loss_epoch, self.best_psnr, self.best_psnr_epoch, self.best_ssim, self.best_ssim_epoch)


    def val(self, test_flag=False):
        PSNR_avg = {}
        SSIM_avg = {}
        Loss_avg = {}
        Loss = toolkits.AverageMeter('Loss')
        Batch_time = toolkits.AverageMeter('batch time')
        PSNR = toolkits.AverageMeter('PSNR')
        SSIM = toolkits.AverageMeter('SSIM')
        crop_border, test_y_channel = self.get_val_status(test_flag)

        # timer
        end = time.time()

        self.model.eval()
        with torch.no_grad():
            loaders = self.test_loader if test_flag else self.val_loader
            for loader_name, loader in loaders.items():
                PSNR.reset()
                SSIM.reset()
                Loss.reset()
                self.logging.info('Phase {} - {}'.format('Test' if test_flag else 'Val', loader_name))
                self.logging.info(f'Crop Border - {crop_border}; Test on Y channel - {test_y_channel}.')
                for batch, (inputs, targets) in enumerate(loader):
                    batch_size = inputs.size(0)
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)

                    loss = self.criterion(outputs, targets)
                    ssim = calc_SSIM(torch.clamp(outputs,0,1), targets,
                                     crop_border=crop_border, test_y_channel=test_y_channel)
                    psnr = calc_PSNR(outputs, targets,
                                     crop_border=crop_border, test_y_channel=test_y_channel)
                    PSNR.update(psnr, batch_size)
                    SSIM.update(ssim, batch_size)
                    Loss.update(loss.item(), batch_size)
                    Batch_time.update(time.time() - end)
                    end = time.time()

                    if self.disp_freq is not None and batch % self.disp_freq == 0:
                        self.logging.info('batch [{0}/{1}]  \t'
                                'Time {Batch_time.val:.3f} ({Batch_time.avg:.3f})\t'
                                'Loss {Loss.val:.3f} ({Loss.avg:.3f})\t'
                                'PSNR {PSNR.val:.3f} ({PSNR.avg:.3f})\t'
                                'SSIM {SSIM.val:.3f} ({SSIM.avg:.3f})\t'
                                .format(batch, len(loader), Batch_time=Batch_time, Loss=Loss,
                                        PSNR=PSNR, SSIM=SSIM))
                PSNR_avg.update({loader_name: PSNR.avg})
                SSIM_avg.update({loader_name: SSIM.avg})
                Loss_avg.update({loader_name: Loss.avg})
        # we assume that there is only one val loader.
        if not test_flag:
            return PSNR.avg, SSIM.avg, Loss.avg
        return PSNR_avg, SSIM_avg, Loss_avg



class IterTrainer(BaseTrainer):
    trainer_type = 'iter'

    def __init__(self, args):
        super(IterTrainer, self).__init__(args)
        self.iters = self.cfg['train_settings']['train_len']
        self.save_freq = self.cfg['train_settings']['save_ckpt_freq']

    def reset_counter(self):
        super(IterTrainer, self).reset_counter()


    def init_training_settings(self):
        super(IterTrainer, self).init_training_settings()
        # TODO: here we fix prefetcher as CPUPrefetcher
        from datasets.prefetch_dataloader import CPUPrefetcher
        self.prefetcher = CPUPrefetcher(self.train_loader)

    def train(self):
        Loss = toolkits.AverageMeter('Loss')
        PSNR = toolkits.AverageMeter('PSNR')
        SSIM = toolkits.AverageMeter('SSIM')
        Batch_time = toolkits.AverageMeter('batch time')

        self.curr_iter = self.start_iter
        self.curr_epoch = self.start_epoch

        # model state
        self.model.train()
        self.logging.info('<-Training Phase->')
        self.logging.info('Epoch [%d/%d]  lr: %e', self.curr_epoch, self.args.total_epochs,
                          self.optimizer.state_dict()['param_groups'][0]['lr'])

        while self.curr_iter <= self.iters:
            if self.args.gpu_flag:
                # TODO: check this
                self.train_loader.sampler.set_epoch(self.curr_epoch)
                self.prefetcher.reset()
                end = time.time()
                self.train_data = self.prefetcher.next()

                while self.train_data is not None:
                    self.model.train()
                    inputs, targets = self.train_data['lq'], self.train_data['gt']
                    batch_size = inputs.size(0)
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    self.curr_iter += 1
                    if self.curr_iter > self.iters:
                        break
                    # update learning rate 
                    self.update_learning_rate(self.curr_iter)
                    # forward and backward
                    outputs = self.model(inputs) # 0.01s
                    loss = self.criterion(outputs, targets)
                    self.optimizer.zero_grad()
                    loss.backward() # 0.02s
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    self.optimizer.step() # 0.01s

                    with torch.no_grad():
                        psnr = calc_PSNR(outputs, targets)
                        ssim = calc_SSIM(torch.clamp(outputs,0,1), targets)
                    # display
                    Loss.update(loss.item(), batch_size)
                    Batch_time.update(time.time() - end)
                    PSNR.update(psnr, batch_size)
                    SSIM.update(ssim, batch_size)

                    if self.disp_freq is not None and self.curr_iter % self.disp_freq == 0:
                        self.logging.info('batch [{0}/{1}] \t'
                            'Time {Batch_time.val:.3f} ({Batch_time.avg:.3f})\t'
                            'Loss {Loss.val:.3f} ({Loss.avg:.3f})\t'
                            'PSNR {PSNR.val:.3f} ({PSNR.avg:.3f})\t'
                            'SSIM {SSIM.val:.3f} ({SSIM.avg:.3f})'
                            .format(self.curr_iter, self.iters, Batch_time=Batch_time, Loss=Loss,
                            PSNR=PSNR, SSIM=SSIM))

                    # save model
                    if self.curr_iter % self.save_freq == 0 and not self.slave:
                        self.logging.info(f'Saving models and training states at iter {self.curr_iter}.')

                        # validation
                        self.logging.info('<-Validating Phase->')
                        psnr_val, ssim_val, loss_val = self.val()

                        # model save
                        if psnr_val > self.best_psnr and not math.isinf(psnr_val):
                            self.best_psnr = psnr_val
                            self.best_psnr_iter = self.curr_iter
                            self.best_psnr_epoch = self.curr_epoch
                        if ssim_val > self.best_ssim:
                            self.best_ssim = ssim_val
                            self.best_ssim_iter = self.curr_iter
                            self.best_ssim_epoch = self.curr_epoch
                        if loss_val < self.best_loss:
                            self.best_loss = loss_val
                            self.best_loss_iter = self.curr_iter
                            self.best_loss_epoch = self.curr_epoch
                        if self.args.save_flag:
                            toolkits.save_checkpoint({
                                'iter': self.curr_iter,
                                'epoch': self.curr_epoch,
                                'state_dict': self.model.state_dict() if not self.args.multi_cards else self.model.module.state_dict(),
                                'psnr': psnr_val,
                                'ssim': ssim_val,
                                'loss': loss_val,
                                'optimizer': self.optimizer.state_dict()}, [f'{self.curr_iter}.pth.tar'], self.args.save_path)
                        self.logging.info('PSNR:%4f SSIM:%4f Loss:%4f / Best_PSNR:%4f Best_SSIM:%4f Best_Loss:%4f',
                        psnr_val, ssim_val, loss_val, self.best_psnr, self.best_ssim, self.best_loss)

                        self.logging.info('<-Training Phase->')

                    end = time.time()
                    # next batch of data
                    self.train_data = self.prefetcher.next()

                # end of one epoch
                self.curr_epoch += 1
                self.logging.info('<-Training Phase->')
                self.logging.info('Epoch [%d/%d]  lr: %e', self.curr_epoch, self.args.total_epochs,
                                                           self.optimizer.state_dict()['param_groups'][0]['lr'])


        # end of the whole training
        self.logging.info('<-Training Finished->')
        self.logging.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)',
                 self.best_loss, self.best_loss_epoch, self.best_psnr, self.best_psnr_epoch, self.best_ssim, self.best_ssim_epoch)


    def val(self, test_flag=False):
        PSNR_avg = {}
        SSIM_avg = {}
        Loss_avg = {}
        Loss = toolkits.AverageMeter('Loss')
        Batch_time = toolkits.AverageMeter('batch time')
        PSNR = toolkits.AverageMeter('PSNR')
        SSIM = toolkits.AverageMeter('SSIM')
        crop_border, test_y_channel = self.get_val_status(test_flag)

        # timer
        end = time.time()

        self.model.eval()
        with torch.no_grad():
            loaders = self.test_loader if test_flag else self.val_loader
            for loader_name, loader in loaders.items():
                PSNR.reset()
                SSIM.reset()
                Loss.reset()
                self.logging.info('Phase {} - {}'.format('Test' if test_flag else 'Val', loader_name))
                self.logging.info(f'Crop Border - {crop_border}; Test on Y channel - {test_y_channel}.')
                for batch, data in enumerate(loader):
                    inputs, targets = data['lq'].to(self.device), data['gt'].to(self.device)
                    batch_size = inputs.size(0)
                    outputs = self.model(inputs)

                    loss = self.criterion(outputs, targets)
                    ssim = calc_SSIM(torch.clamp(outputs,0,1), targets,
                                     crop_border=crop_border, test_y_channel=test_y_channel)
                    psnr = calc_PSNR(outputs, targets,
                                     crop_border=crop_border, test_y_channel=test_y_channel)
                    PSNR.update(psnr, batch_size)
                    SSIM.update(ssim, batch_size)
                    Loss.update(loss.item(), batch_size)
                    Batch_time.update(time.time() - end)
                    end = time.time()
                    if self.disp_freq is not None and batch % self.disp_freq == 0:
                        self.logging.info('batch [{0}/{1}]  \t'
                                'Time {Batch_time.val:.3f} ({Batch_time.avg:.3f})\t'
                                'Loss {Loss.val:.3f} ({Loss.avg:.3f})\t'
                                'PSNR {PSNR.val:.3f} ({PSNR.avg:.3f})\t'
                                'SSIM {SSIM.val:.3f} ({SSIM.avg:.3f})\t'
                                .format(batch, len(loader), Batch_time=Batch_time, Loss=Loss,
                                        PSNR=PSNR, SSIM=SSIM))

                PSNR_avg.update({loader_name: PSNR.avg})
                SSIM_avg.update({loader_name: SSIM.avg})
                Loss_avg.update({loader_name: Loss.avg})
        # we assume that there is only one val loader.
        if not test_flag:
            return PSNR.avg, SSIM.avg, Loss.avg

        return PSNR_avg, SSIM_avg, Loss_avg



