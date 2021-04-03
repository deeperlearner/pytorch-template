import os
from abc import abstractmethod

import torch
import numpy as np

from base import Cross_Valid
from logger import get_logger, TensorboardWriter
from utils import msg_box


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, torch_args: dict, save_dir, **kwargs):
        # data_loaders
        self.train_data_loaders = torch_args['data_loaders']['train']
        self.valid_data_loaders = torch_args['data_loaders']['valid']
        # models
        self.models = torch_args['models']
        # losses
        self.losses = torch_args['losses']
        # metrics
        self.metrics_iter = torch_args['metrics']['iter']
        self.metrics_epoch = torch_args['metrics']['epoch']
        # optimizers
        self.optimizers = torch_args['optimizers']
        # lr_schedulers
        self.lr_schedulers = torch_args['lr_schedulers']

        self.model_dir = save_dir['model']
        # set json kwargs to self.{kwargs}
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.logger = get_logger('trainer', verbosity=self.verbosity)
        if self.early_stop <= 0 or self.early_stop is None:
            self.early_stop = np.inf
        self.start_epoch = 1

        # configuration to monitor model performance and save best
        self.num_best = 0
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf

        # setup visualization writer instance
        self.writer = TensorboardWriter(save_dir['log'], self.logger, self.tensorboard)

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
                    log_best = log_mean
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0 or best:
                self.logger.info("Best {}: {:.5f}".format(self.mnt_metric, self.mnt_best))
                self._save_checkpoint(epoch, save_best=best)

        return log_best

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'models': {key: value.state_dict() for key, value in self.models.items()},
            'optimizers': {key: value.state_dict() for key, value in self.optimizers.items()},
            'monitor_best': self.mnt_best,
        }
        k_fold = Cross_Valid.k_fold
        fold_idx = Cross_Valid.fold_idx
        fold_prefix = f"fold_{fold_idx}_" if k_fold > 1 else ''

        if save_best:
            if self.save_the_best:
                filename = str(self.model_dir / f"{fold_prefix}model_best.pth")
            else:
                self.num_best += 1
                filename = str(self.model_dir / f"{fold_prefix}model_best{self.num_best}.pth")
        else:
            filename = str(self.model_dir / f"{fold_prefix}checkpoint-epoch{epoch}.pth")
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
