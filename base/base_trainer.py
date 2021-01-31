import os
from abc import abstractmethod

import torch
import numpy as np

from base import BaseDataLoader
import data_loader
from logger import TensorboardWriter
import model as module_arch
import model.loss as module_loss
import model.metric as module_metric
from utils import get_by_path


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config):
        self.config = config
        # trainer keyword arguments
        cfg_trainer = config['trainer']['kwargs']
        self.finetune = cfg_trainer['finetune']
        self.epochs = cfg_trainer['epochs']
        self.len_epoch = cfg_trainer.get('len_epoch', None)
        self.save_period = cfg_trainer['save_period']
        self.save_the_best = cfg_trainer['save_the_best']
        verbosity = cfg_trainer['verbosity']
        monitor = cfg_trainer.get('monitor', 'off')
        self.early_stop = cfg_trainer.get('early_stop', np.inf)
        if self.early_stop <= 0 or self.early_stop is None:
            self.early_stop = np.inf
        tensorboard = cfg_trainer['tensorboard']

        self.logger = config.get_logger('trainer', verbosity)
        # datasets
        self.train_datasets = dict()
        self.valid_datasets = dict()
        ## train
        keys = ['datasets', 'train']
        for name in get_by_path(config, keys):
            self.train_datasets[name] = config.init_obj([*keys, name], 'data_loader')
        ## valid
        valid = False
        keys = ['datasets', 'valid']
        for name in get_by_path(config, keys):
            valid = True
            self.valid_datasets[name] = config.init_obj([*keys, name], 'data_loader')

        # data_loaders
        self.n_fold = config['data_loaders']['N_fold']
        self.train_data_loaders = dict()
        self.valid_data_loaders = dict()
        ## train
        keys = ['data_loaders', 'train']
        for name in get_by_path(config, keys):
            dataset = self.train_datasets[name]
            self.train_data_loaders[name] = config.init_obj([*keys, name], 'data_loader', dataset, N_fold=self.n_fold)
            if not valid:
                self.valid_data_loaders[name] = self.train_data_loaders[name].valid_loader
        ## valid
        keys = ['data_loaders', 'valid']
        for name in get_by_path(config, keys):
            dataset = self.valid_datasets[name]
            self.valid_data_loaders[name] = config.init_obj([*keys, name], 'data_loader', dataset)

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])
        # models
        self.models = dict()
        logger_model = config.get_logger('model', verbosity=1)
        for name in config['models']:
            model = config.init_obj(['models', name], 'model')
            logger_model.info(model)
            model = model.to(self.device)
            if len(self.device_ids) > 1:
                model = torch.nn.DataParallel(model, device_ids=self.device_ids)
            self.models[name] = model

        # losses
        self.losses = dict()
        for name in config['losses']:
            self.losses[name] = config.init_ftn(['losses', name], module_loss)

        # metrics
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        # optimizers
        self.optimizers = dict()
        for name in config['optimizers']:
            trainable_params = filter(lambda p: p.requires_grad, self.models[name].parameters())
            self.optimizers[name] = config.init_obj(['optimizers', name], torch.optim, trainable_params)

        # learning rate schedulers
        self.lr_schedulers = dict()
        for name in config['lr_schedulers']:
            self.lr_schedulers[name] = config.init_obj(['lr_schedulers', name],
                                                        torch.optim.lr_scheduler, self.optimizers[name])
        # configuration to monitor model performance and save best
        self.mnt_bests = np.zeros(self.n_fold)
        self.num_best = 0
        if monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir['model']

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.save_dir['log'], self.logger, tensorboard)

    def cross_valid(self):
        # cross validation data
        for name, loader in self.train_data_loaders.items():
            # reconstruct train/valid data
            if self.config['data_loaders']['train'][name]['split_valid']:
                keys = ['data_loaders', 'train', name]
                dataset = self.train_datasets[name]
                self.train_data_loaders[name] = self.config.init_obj(keys, 'data_loader',
                                                                    dataset, N_fold=self.n_fold)
                self.valid_data_loaders[name] = loader.valid_loader
        # re-initialize model weight
        for model in self.models.values():
            model.weights_init()

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            train_log = self._train_epoch(epoch)
            log_mean = train_log['mean']

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance strictly improved or not, according to mnt_metric
                    improved = (self.mnt_mode == 'min' and log_mean[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log_mean[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log_mean[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0 or best:
                self.logger.info('Best {}: {}'.format(self.mnt_metric, self.mnt_best))
                self._save_checkpoint(epoch, save_best=best)

        # cross validation is enabled
        if self.n_fold > 1:
            fold_idx = BaseDataLoader.fold_idx
            # record the result of each cross validation
            self.mnt_bests[fold_idx-1] = self.mnt_best
            if fold_idx == self.n_fold:
                # record the average result of cross validation
                cv_monitor = self.mnt_bests.mean()
                self.logger.info(f'{self.n_fold}-fold cross validation of {self.mnt_metric}: {cv_monitor}')
            # do cross validation
            if fold_idx < self.n_fold:
                self.cross_valid()

    def _prepare_device(self, n_gpu_use):
        """
        Setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'arch': self.config['name'],
            'epoch': epoch,
            'models': {key: value.state_dict() for key, value in self.models.items()},
            'optimizers': {key: value.state_dict() for key, value in self.optimizers.items()},
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            if self.save_the_best:
                filename = str(self.checkpoint_dir / 'model_best.pth')
            else:
                self.num_best += 1
                filename = str(self.checkpoint_dir / f'model_best{self.num_best}.pth')
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving model: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path, finetune=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        if not finetune:
            # resume training
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']

        # load each model params from checkpoint.
        for key, value in checkpoint['models'].items():
            try:
                self.models[key].load_state_dict(value)
            except KeyError:
                print("models not match, can not resume.")

        # load each optimizer from checkpoint.
        for key, value in checkpoint['optimizers'].items():
            try:
                self.optimizers[key].load_state_dict(value)
            except KeyError:
                print("optimizers not match, can not resume.")

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
