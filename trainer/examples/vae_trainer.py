import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from utils import inf_loop, MetricTracker


class VAE_Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config, len_epoch=None):
        super().__init__(config)

        # dataloader
        self.data_loader = config.init_obj('data_loader', module_data, mode=config.mode)
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
        self.model = config.init_obj('arch', module_arch)
        self.model = self.model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        self.logger.info(self.model)

        # get function handles of loss and metrics
        self.criterion = config.init_ftn('loss', module_loss)
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
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output, mu, logvar = self.model(data)
            loss, MSE, KLD = self.criterion(output, mu, logvar, data)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('MSE', MSE.item())
            self.writer.add_scalar('KLD', KLD.item())
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
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                output, mu, logvar = self.model(data)
                loss, MSE, KLD = self.criterion(output, mu, logvar, data)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metrics:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            filename = str(self.checkpoint_dir / 'model_best.pth')
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving model: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
