from abc import abstractmethod

import torch
from numpy import inf

import data_loader as module_data
from logger import TensorboardWriter
import model as module_arch
import model.loss as module_loss
import model.metric as module_metric


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config):
        self.config = config
        # trainer
        cfg_trainer = config['trainer']['args']
        self.epochs = cfg_trainer['epochs']
        self.len_epoch = cfg_trainer['len_epoch']
        self.save_period = cfg_trainer['save_period']
        verbosity = cfg_trainer['verbosity']
        monitor = cfg_trainer.get('monitor', 'off')
        self.early_stop = cfg_trainer.get('early_stop', inf)
        tensorboard = cfg_trainer['tensorboard']

        # datasets
        self.datasets = dict()
        for name in config['datasets']:
            self.datasets[name] = config.init_obj('datasets', name, module_data, mode=config.mode)

        # data_loaders
        self.data_loaders = dict()
        for name in config['data_loaders']:
            dataset = self.datasets[name]
            self.data_loaders[name] = config.init_obj('data_loaders', name, module_data, dataset)

        self.logger = config.get_logger('trainer', verbosity)
        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])
        # models
        self.models = dict()
        for name in config['models']:
            self.models[name] = config.init_obj('models', name, module_arch)
            self.models[name] = self.models[name].to(self.device)
            if len(self.device_ids) > 1:
                self.models[name] = torch.nn.DataParallel(self.models[name], device_ids=self.device_ids)
            self.logger.info(self.models[name])

        # losses
        self.losses = dict()
        for name in config['losses'].keys():
            self.losses[name] = config.init_ftn('losses', name, module_loss)

        # metrics
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        # optimizers
        self.optimizers = dict()
        for name in config['optimizers']:
            trainable_params = filter(lambda p: p.requires_grad, self.models[name].parameters())
            self.optimizers[name] = config.init_obj('optimizers', name, torch.optim, trainable_params)

        # learning rate schedulers
        self.lr_schedulers = dict()
        for name in config['lr_schedulers']:
            self.lr_schedulers[name] = config.init_obj('lr_schedulers', name,
                                                        torch.optim.lr_scheduler, self.optimizers[name])

        # configuration to monitor model performance and save best
        if monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir['model']

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.save_dir['log'], self.logger, tensorboard)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

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
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self.logger.info('\nMetric: {}'.format(log))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0 or best:
                self.logger.info('Best val loss: {}'.format(self.mnt_best))
                self._save_checkpoint(epoch, save_best=best)

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
            'models': { key: value.state_dict() for key, value in self.models.items() },
            'optimizers': { key: value.state_dict() for key, value in self.optimizers.items() },
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

        # load each model params from checkpoint.
        for key in self.models:
            try:
                self.models[key].load_state_dict(checkpoint['models'][key])
            except KeyError:
                print("models not match, can not resume.")

        # load each optimizer from checkpoint.
        for key in self.optimizers:
            try:
                self.optimizers[key].load_state_dict(checkpoint['optimizers'][key])
            except KeyError:
                print("optimizers not match, can not resume.")

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
