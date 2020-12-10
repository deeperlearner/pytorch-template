import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from utils import inf_loop, MetricTracker


class GAN_Trainer(BaseTrainer):
    """
    GAN trainer class
    """
    def __init__(self, config, len_epoch=None):
        super().__init__(config)

        # dataloader
        self.data_loader = self.data_loaders['data']
        self.valid_data_loader = self.data_loader.valid_loader
        self.do_validation = self.valid_data_loader is not None
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(self.data_loader)
            self.len_epoch = len_epoch
        self.log_step = int(np.sqrt(self.data_loader.batch_size))

        # model
        self.model_G = config.init_obj('net_G', module_arch)
        self.model_D = config.init_obj('net_D', module_arch)
        self.model_G = self.model_G.to(self.device)
        self.model_D = self.model_D.to(self.device)
        if len(self.device_ids) > 1:
            self.model_G = torch.nn.DataParallel(self.model_G, device_ids=self.device_ids)
            self.model_D = torch.nn.DataParallel(self.model_D, device_ids=self.device_ids)
        self.logger.info(self.model_G)
        self.logger.info(self.model_D)
        self.model_G.apply(module_arch.weights_init)
        self.model_D.apply(module_arch.weights_init)

        # get function handles of loss and metrics
        self.criterion = config.init_ftn('loss', module_loss)
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params_G = filter(lambda p: p.requires_grad, self.model_G.parameters())
        trainable_params_D = filter(lambda p: p.requires_grad, self.model_D.parameters())
        self.optimizer_G = config.init_obj('optimizer', torch.optim, trainable_params_G)
        self.optimizer_D = config.init_obj('optimizer', torch.optim, trainable_params_D)

        self.lr_scheduler_G = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer_G)
        self.lr_scheduler_D = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer_D)

        self.train_metrics = MetricTracker('loss_G', 'loss_D', *[m.__name__ for m in self.metrics], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model_G.train()
        self.model_D.train()
        self.train_metrics.reset()
        real_label, fake_label = 1, 0
        for batch_idx, (data, _) in enumerate(self.data_loader):
            """
            (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            """
            # Train with all-real batch
            self.optimizer_D.zero_grad()
            data = data.to(self.device)
            batch_size = data.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
            output = self.model_D(data).view(-1)
            loss_D_real = self.criterion(output, label)
            loss_D_real.backward()
            self.optimizer_D.step()

            # Train with all-fake batch
            noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
            fake = self.model_G(noise)
            label.fill_(fake_label)
            output = self.model_D(fake.detach()).view(-1)
            loss_D_fake = self.criterion(output, label)
            loss_D_fake.backward()
            loss_D = loss_D_real + loss_D_fake
            # Update D
            self.optimizer_D.step()

            """
            (2) Update G network: maximize log(D(G(z)))
            """
            self.optimizer_G.zero_grad()
            label.fill_(real_label)
            output = self.model_D(fake).view(-1)
            loss_G = self.criterion(output, label)
            loss_G.backward()
            # Update G
            self.optimizer_G.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss_G', loss_G.item())
            self.train_metrics.update('loss_D', loss_D.item())
            #for met in self.metrics:
            #    self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss_G: {:.6f} Loss_D: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_G.item(),
                    loss_D.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            pass
            #val_log = self._valid_epoch(epoch)
            #log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        return

    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)
