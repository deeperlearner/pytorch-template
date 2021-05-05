import os
import argparse
import collections

import torch
from sklearn.utils.class_weight import compute_class_weight
try:    
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/NVIDIA/apex.")
import optuna

from base import Cross_Valid
from logger import get_logger
import models.loss as module_loss
import models.metric as module_metric
from parse_config import ConfigParser
from utils import ensure_dir, prepare_device, set_by_path, get_by_path, msg_box


def main():
    k_fold = config['trainer'].get('k_fold', 1)
    fold_idx = config['trainer'].get('fold_idx', 0)

    if fold_idx > 0:
        # do on fold_idx, which is for multiprocessing cross validation
        # if multiprocessing, turn off debug logging to avoid messing up stdout
        config['trainer']['kwargs']['verbosity'] = 1
        verbosity = 1
        k_loop = 1
    else:
        # do full cross validation in single thread
        verbosity = 2
        k_loop = k_fold

    logger = get_logger('train', verbosity=verbosity)
    train_msg = msg_box("TRAIN")
    logger.debug(train_msg)

    # setup GPU device if available, move model into configured device
    device, device_ids = prepare_device(config['n_gpu'])

    # datasets
    train_datasets = dict()
    valid_datasets = dict()
    ## train
    keys = ['datasets', 'train']
    for name in get_by_path(config, keys):
        train_datasets[name] = config.init_obj([*keys, name], 'data_loaders')
    ## valid
    valid_exist = False
    keys = ['datasets', 'valid']
    for name in get_by_path(config, keys):
        valid_exist = True
        valid_datasets[name] = config.init_obj([*keys, name], 'data_loaders')

    # losses
    losses = dict()
    for name in config['losses']:
        kwargs = {}
        # TODO
        if config['losses'][name].get('balanced', False):
            target = train_datasets['data'].y_train
            weight = compute_class_weight(class_weight='balanced',
                                          classes=target.unique(),
                                          y=target)
            weight = torch.FloatTensor(weight).to(device)
            kwargs.update(pos_weight=weight[1])
        losses[name] = config.init_obj(['losses', name], module_loss, **kwargs)

    # metrics
    metrics_iter = [getattr(module_metric, met) for met in config['metrics']['per_iteration']]
    metrics_epoch = [getattr(module_metric, met) for met in config['metrics']['per_epoch']]

    # unchanged objects in each fold
    torch_args = {'datasets': {'train': train_datasets, 'valid': valid_datasets},
                  'losses': losses,
                  'metrics': {'iter': metrics_iter, 'epoch': metrics_epoch}}

    if k_fold > 1:  # cross validation enabled
        train_datasets['data'].split_cv_indexes(k_fold)
    Cross_Valid.create_CV(k_fold, fold_idx)

    for k in range(k_loop):
        # data_loaders
        train_data_loaders = dict()
        valid_data_loaders = dict()
        ## train
        keys = ['data_loaders', 'train']
        for name in get_by_path(config, keys):
            dataset = train_datasets[name]
            train_data_loaders[name] = config.init_obj([*keys, name], 'data_loaders', dataset)
            if not valid_exist:
                valid_data_loaders[name] = train_data_loaders[name].valid_loader
        ## valid
        keys = ['data_loaders', 'valid']
        for name in get_by_path(config, keys):
            dataset = valid_datasets[name]
            valid_data_loaders[name] = config.init_obj([*keys, name], 'data_loaders', dataset)

        # models
        models = dict()
        logger_model = get_logger('model', verbosity=1)
        for name in config['models']:
            model = config.init_obj(['models', name], 'models')
            logger_model.info(model)
            model = model.to(device)
            if len(device_ids) > 1:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
            models[name] = model

        # optimizers
        optimizers = dict()
        for name in config['optimizers']:
            trainable_params = filter(lambda p: p.requires_grad, models[name].parameters())
            optimizers[name] = config.init_obj(['optimizers', name], torch.optim, trainable_params)

        # learning rate schedulers
        lr_schedulers = dict()
        for name in config['lr_schedulers']:
            lr_schedulers[name] = config.init_obj(['lr_schedulers', name],
                                                  torch.optim.lr_scheduler, optimizers[name])

        # update objects for each fold
        update_args = {'data_loaders': {'train': train_data_loaders, 'valid': valid_data_loaders},
                       'models': models,
                       'optimizers': optimizers,
                       'lr_schedulers': lr_schedulers,
                       'amp': None}

        # amp
        if config['trainer']['kwargs'].get('apex', False):
            # TODO: revise here if multiple models and optimizers
            models['model'], optimizers['model'] = amp.initialize(
                models['model'], optimizers['model'], opt_level='O1')
            update_args['amp'] = amp

        torch_args.update(update_args)
        trainer = config.init_obj(['trainer'], 'trainers', torch_args,
                                  config.save_dir, config.resume, device)
        log_best = trainer.train()

        # cross validation
        if k_fold > 1:
            idx = Cross_Valid.fold_idx
            save_path = config.save_dir['metrics_best'] / f"fold_{idx}.pkl"
            log_best.to_pickle(save_path)
            Cross_Valid.next_fold()
        else:
            msg = msg_box("result")
            logger.info(f"{msg}\n{log_best}")

    return log_best[mnt_metric]


def objective(trial):
    # TODO: hyperparameters search spaces
    search_opt_type = trial.suggest_categorical("optimizer", ['Adam', 'RMSprop', 'SGD'])
    set_by_path(config, "optimizers;model;type", search_opt_type)
    search_lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    set_by_path(config, "optimizers;model;kwargs;lr", search_lr)

    result = main()
    return result


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='training')
    run_args = args.add_argument_group('run_args')
    run_args.add_argument('-c', '--config', default="configs/config.json", type=str)
    run_args.add_argument('--resume', default=None, type=str)
    run_args.add_argument('--mode', default='train', type=str)
    run_args.add_argument('--run_id', default=None, type=str)
    run_args.add_argument('--log_name', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    mod_args = args.add_argument_group('mod_args')
    CustomArgs = collections.namedtuple('CustomArgs', "flags type target")
    options = [
        CustomArgs(['--fold_idx'], type=int, target="trainer;fold_idx"),  # fold_idx > 0 means multiprocessing is enabled
        CustomArgs(['--num_workers'], type=int, target="data_loaders;train;data;kwargs;DataLoader_kwargs;num_workers"),
        CustomArgs(['--lr', '--learning_rate'], type=float, target="optimizers;model;args;lr"),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target="data_loaders;train;data;args;DataLoader_kwargs;batch_size"),
    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, type=opt.type)

    config = ConfigParser.from_args(args, options)

    if config['optuna']:
        max_min, mnt_metric = config['trainer']['kwargs']['monitor'].split()
        study = optuna.create_study(direction='maximize' if max_min == 'max' else 'minimize')
        study.optimize(objective, n_trials=10)
    else:
        main()
