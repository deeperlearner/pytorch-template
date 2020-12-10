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

class ADDA_Pretrain(BaseTrainer):
    """
    Pretrain trainer for ADDA
    """
    def __init__(self, config, len_epoch=None):
        super().__init__(config)
        self.logger.info("Pretraining")
        # dataloaders
        self.source_loader = config.init_obj('source_loader', module_data, mode=config.mode, validation_split=0.2)
        self.valid_loader = self.source_loader.valid_loader
        self.do_validation = self.valid_loader is not None
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.source_loader)
        else:
            # iteration-based training
            self.source_loader = inf_loop(self.source_loader)
            self.len_epoch = len_epoch
        self.log_step = int(np.sqrt(self.len_epoch))

        # models
        self.models_init(['Encoder', 'Classifier'], module_arch)
        self.encoder = self.models['Encoder']
        self.encoder.apply(module_arch.weights_init)
        self.classfier = self.models['Classifier']
        self.classfier.apply(module_arch.weights_init)

        self.criterion = config.init_ftn('loss', module_loss)
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        trainable_params = filter(lambda p: p.requires_grad, list(self.encoder.parameters()) + list(self.classfier.parameters()))
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)

    def _train_epoch(self, epoch):
        self.encoder.train()
        self.classfier.train()
        for batch_idx, (_, source_data, source_label) in enumerate(self.source_loader):
            self.optimizer.zero_grad()
            source_data, source_label = source_data.to(self.device), source_label.to(self.device)

            pred = self.classfier(self.encoder(source_data))
            loss = self.criterion(pred, source_label)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metrics:
                self.train_metrics.update(met.__name__, met(pred, source_label))

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

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.encoder.eval()
        self.classfier.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (_, data, label) in enumerate(self.valid_loader):
                data, label = data.to(self.device), label.to(self.device)
                output = self.classfier(self.encoder(data))
                loss = self.criterion(output, label)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metrics:
                    self.valid_metrics.update(met.__name__, met(output, label))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)

class ADDA_Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config, len_epoch=None):
        super().__init__(config)
        len_epoch = 70000 // config['target_loader']['args']['batch_size']
        pretrainer = ADDA_Pretrain(config)
        pretrainer.train()

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
        self.src_encoder = pretrainer.encoder
        self.clf = pretrainer.classfier
        self.tgt_encoder = pretrainer.encoder
        self.models_init(['Discriminator'], module_arch)
        self.discreiminator = self.models['Discriminator']
        self.discreiminator.apply(module_arch.weights_init)

        self.criterion = config.init_ftn('loss', module_loss)
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        trainable_params = filter(lambda p: p.requires_grad, list(self.tgt_encoder.parameters()))
        self.optimizer_enc = config.init_obj('optimizer', torch.optim, trainable_params)
        trainable_params = filter(lambda p: p.requires_grad, list(self.discreiminator.parameters()))
        self.optimizer_dis = config.init_obj('optimizer', torch.optim, trainable_params)

        self.train_metrics = MetricTracker('loss_dis', 'loss_tgt', *[m.__name__ for m in self.metrics], writer=self.writer)
        self.valid_metrics = MetricTracker('loss_tgt', *[m.__name__ for m in self.metrics], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.tgt_encoder.train()
        self.discreiminator.train()

        self.train_metrics.reset()
        source_id, target_id = 1, 0
        for batch_idx, ((_, source_data, source_label), (_, target_data, target_label)) in enumerate(zip(self.source_loader, cycle(self.target_loader))):
            ######################
            # train discreiminator
            ######################
            source_data, source_label = source_data.to(self.device), source_label.to(self.device)
            target_data, target_label = target_data.to(self.device), target_label.to(self.device)

            self.optimizer_dis.zero_grad()

            feat_src = self.src_encoder(source_data)
            feat_tgt = self.tgt_encoder(target_data)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            pred_concat = self.discreiminator(feat_concat)

            batch_size = source_data.size(0)
            label_src = torch.full((batch_size,), source_id, dtype=torch.long, device=self.device)
            batch_size = target_data.size(0)
            label_tgt = torch.full((batch_size,), target_id, dtype=torch.long, device=self.device)

            label_concat = torch.cat((label_src, label_tgt), 0)

            loss_dis = self.criterion(pred_concat, label_concat)
            loss_dis.backward()
            self.optimizer_dis.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss_dis', loss_dis.item())
            ######################
            # train target encoder
            ######################
            self.optimizer_dis.zero_grad()
            self.optimizer_enc.zero_grad()

            feat_tgt = self.tgt_encoder(target_data)
            pred_tgt = self.discreiminator(feat_tgt)
            
            batch_size = target_data.size(0)
            label_tgt = torch.full((batch_size,), source_id, dtype=torch.long, device=self.device)

            loss_tgt = self.criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            self.optimizer_enc.step()

            self.train_metrics.update('loss_tgt', loss_tgt.item())
            with torch.no_grad():
                self.clf.eval()
                pred = self.clf(feat_tgt)
                for met in self.metrics:
                    self.train_metrics.update(met.__name__, met(pred_concat, label_concat))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_tgt.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.tgt_encoder.eval()
        self.discreiminator.eval()
        self.clf.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (_, target_data, target_label) in enumerate(self.target_valid_loader):
                target_data, target_label = target_data.to(self.device), target_label.to(self.device)
                output = self.tgt_encoder(target_data)
                output = self.clf(output)
                loss = self.criterion(output, target_label)

                self.valid_metrics.update('loss_tgt', loss.item())

                for met in self.metrics:
                    self.valid_metrics.update(met.__name__, met(output, target_label))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)
