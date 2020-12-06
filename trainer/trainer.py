import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config):
        super().__init__(config)

        # data_loaders
        self.data_loader = self.data_loaders['data']
        self.valid_data_loader = self.data_loader.valid_loader
        self.do_validation = self.valid_data_loader is not None
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(self.data_loader)
            self.len_epoch = self.len_epoch
        self.log_step = int(np.sqrt(self.data_loader.batch_size))

        # models
        self.model = self.models['model']

        # losses
        self.criterion = self.losses['model']

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
        for batch_idx, (data, target) in enumerate(self.data_loader):
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
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.do_lr_scheduling:
            self.lr_scheduler.step()
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
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metrics:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)
