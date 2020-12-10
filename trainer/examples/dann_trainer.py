from itertools import cycle

import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from utils import inf_loop, MetricTracker


class DANN_Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config, len_epoch=None):
        len_epoch = 70000 // config['target_loader']['args']['batch_size']
        super().__init__(config)

        # dataloader
        self.source_loader = config.init_obj('source_loader', module_data, mode=config.mode)
        self.target_loader = config.init_obj('target_loader', module_data, mode=config.mode)
        self.target_valid_loader = self.target_loader.valid_loader
        self.do_validation = self.target_valid_loader is not None
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.source_loader)
        else:
            # iteration-based training
            self.source_loader = inf_loop(self.source_loader)
            self.len_epoch = len_epoch
        self.log_step = int(np.sqrt(self.len_epoch))

        # model
        self.models_init(['arch'], module_arch)
        self.model = self.models['arch']

        # get function handles of loss and metrics
        self.class_criterion = config.init_ftn('loss', module_loss)
        self.domain_criterion = config.init_ftn('loss', module_loss)
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        source_id, target_id = 0, 1
        for batch_idx, ((_, source_data, source_label), (_, target_data, _)) in enumerate(zip(self.source_loader, cycle(self.target_loader))):
            p = float(batch_idx + self.len_epoch * epoch) / (self.len_epoch * self.epochs)
            constant = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()
            source_data, source_label, target_data = source_data.to(self.device), source_label.to(self.device), target_data.to(self.device)

            batch_size = source_data.size(0)
            domain_label = torch.full((batch_size,), source_id, dtype=torch.long, device=self.device)

            cls_output, dom_output = self.model(source_data, constant)
            label_loss = self.class_criterion(cls_output, source_label)
            source_domain_loss = self.domain_criterion(dom_output, domain_label)

            batch_size = target_data.size(0)
            domain_label = torch.full((batch_size,), target_id, dtype=torch.long, device=self.device)
            _, dom_output = self.model(target_data, constant)
            target_domain_loss = self.domain_criterion(dom_output, domain_label)

            loss = label_loss + source_domain_loss + target_domain_loss
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metrics:
                self.train_metrics.update(met.__name__, met(cls_output, source_label))

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

        if self.lr_scheduler is not None:
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
            for batch_idx, (_, target_data, target_label) in enumerate(self.target_valid_loader):
                target_data, target_label = target_data.to(self.device), target_label.to(self.device)
                output, _ = self.model(target_data, 0)
                loss = self.class_criterion(output, target_label)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metrics:
                    self.valid_metrics.update(met.__name__, met(output, target_label))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)
