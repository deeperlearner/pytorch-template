import os

import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from model.metric import MetricTracker
from utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config):
        super().__init__(config)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, finetune=self.finetune)

        # data_loaders
        self.train_loader = self.train_data_loaders['data']
        self.valid_loader = self.valid_data_loaders['data']
        self.do_validation = self.valid_loader is not None
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_loader)
        else:
            # iteration-based training
            self.train_loader = inf_loop(self.train_loader)
        self.log_step = int(np.sqrt(self.train_loader.batch_size))

        # models
        self.model = self.models['model']

        # losses
        self.criterion = self.losses['loss']

        # metrics
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)

        # optimizers
        self.optimizer = self.optimizers['model']

        # learning rate schedulers
        self.do_lr_scheduling = len(self.lr_schedulers) > 0
        self.lr_scheduler = self.lr_schedulers['model']

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metrics:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                epoch_debug = f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                current_metrics = self.train_metrics.current()
                metrics_debug = ', '.join(f"{key}: {value:.6f}" for key, value in current_metrics.items())
                self.logger.debug(epoch_debug + metrics_debug)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        train_log = self.train_metrics.result()

        if self.do_validation:
            valid_log = self._valid_epoch(epoch)
            valid_log.set_index('val_' + valid_log.index.astype(str), inplace=True)

        if self.do_lr_scheduling:
            self.lr_scheduler.step()

        log = pd.concat([train_log, valid_log])
        epoch_log = {'epochs': epoch, 'iterations': self.len_epoch * epoch}
        epoch_info = ', '.join(f"{key}: {value}" for key, value in epoch_log.items())
        self.logger.info(f'{epoch_info}\n{log}')

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outputs = torch.FloatTensor().to(self.device)
            targets = torch.FloatTensor().to(self.device)
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                outputs = torch.cat((outputs, output))
                targets = torch.cat((targets, target))

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

            for met in self.metrics:
                self.valid_metrics.update(met.__name__, met(outputs, targets))
            #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, bins='auto')

        valid_log = self.valid_metrics.result()

        return valid_log

    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)
