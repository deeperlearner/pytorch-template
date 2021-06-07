import os 
import sys
import argparse
import collections
import time

import torch
from apex import amp
from sklearn.utils.class_weight import compute_class_weight
import optuna

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from logger import get_logger
from mains import Cross_Valid
import models.loss as module_loss
import models.metric as module_metric
from parse_config import ConfigParser
from utils import ensure_dir, prepare_device, set_by_path, get_by_path, msg_box, consuming_time


def main():
    logger = get_logger('train')
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
    ## compute inverse class frequency as class weight
    if config['datasets'].get('imbalanced', False):
        target = train_datasets['data'].y_train  # TODO
        class_weight = compute_class_weight(class_weight='balanced',
                                            classes=target.unique(),
                                            y=target)
        class_weight = torch.FloatTensor(class_weight).to(device)
    else:
        class_weight = None

    # losses
    losses = dict()
    for name in config['losses']:
        kwargs = {}
        if 'balanced' in get_by_path(config, ['losses', name, 'type']):
            kwargs.update(class_weight=class_weight)
        losses[name] = config.init_obj(['losses', name], module_loss, **kwargs)

    # metrics
    metrics_iter = [getattr(module_metric, met) for met in config['metrics']['per_iteration']]
    metrics_epoch = [getattr(module_metric, met) for met in config['metrics']['per_epoch']]
    if 'pick_threshold' in config['metrics']:
        metrics_threshold = config.init_obj(['metrics', 'pick_threshold'], module_metric)
    else:
        metrics_threshold = None

    torch_objs = {'datasets': {'train': train_datasets, 'valid': valid_datasets},
                  'losses': losses,
                  'metrics': {'iter': metrics_iter, 'epoch': metrics_epoch,
                              'threshold': metrics_threshold}
                 }

    if K_FOLD > 1:  # cross validation enabled
        train_datasets['data'].split_cv_indexes(K_FOLD)

    results = []
    Cross_Valid.create_CV(K_FOLD)
    start = time.time()
    for k in range(K_FOLD):
        # data_loaders
        train_data_loaders = dict()
        valid_data_loaders = dict()
        ## train
        keys = ['data_loaders', 'train']
        for name in get_by_path(config, keys):
            kwargs = {}
            if 'imbalanced' in get_by_path(config, [*keys, name, 'module']):
                kwargs.update(
                    class_weight=class_weight.cpu().detach().numpy(),
                    target=target)
            dataset = train_datasets[name]
            loaders = config.init_obj([*keys, name], 'data_loaders', dataset, **kwargs)
            train_data_loaders[name] = loaders.train_loader
            if not valid_exist:
                valid_data_loaders[name] = loaders.valid_loader
        ## valid
        keys = ['data_loaders', 'valid']
        for name in get_by_path(config, keys):
            dataset = valid_datasets[name]
            loaders = config.init_obj([*keys, name], 'data_loaders', dataset)
            valid_data_loaders[name] = loaders.valid_loader

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

        torch_objs.update(
            {'data_loaders': {'train': train_data_loaders,
                              'valid': valid_data_loaders},
             'models': models,
             'optimizers': optimizers,
             'lr_schedulers': lr_schedulers,
             'amp': None}
        )

        # amp
        if config['trainer']['kwargs']['apex']:
            # TODO: revise here if multiple models and optimizers
            models['model'], optimizers['model'] = amp.initialize(
                models['model'], optimizers['model'], opt_level='O1')
            torch_objs['amp'] = amp

        trainer = config.init_obj(['trainer'], 'trainers', torch_objs,
                                  config.save_dir, config.resume, device)
        log_best = trainer.train()
        results.append(log_best[MNT_METRIC])

        # cross validation
        if K_FOLD > 1:
            Cross_Valid.next_fold()

    # result
    msg = msg_box("result")
    logger.info(msg)

    end = time.time()
    total_time = consuming_time(start, end)
    logger.info(f"Consuming time: {total_time}.")

    avg_result = sum(results) / len(results)
    logger.info(f"{MNT_METRIC}: {avg_result:.6f}")

    return avg_result


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
        CustomArgs(['--name'], type=str, target="name"),
        CustomArgs(['--fold_idx'], type=int, target="trainer;fold_idx"),  # fold_idx > 0 means multiprocessing is enabled
        CustomArgs(['--num_workers'], type=int, target="data_loaders;train;data;kwargs;DataLoader_kwargs;num_workers"),
        CustomArgs(['--lr', '--learning_rate'], type=float, target="optimizers;model;args;lr"),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target="data_loaders;train;data;args;DataLoader_kwargs;batch_size"),
    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, type=opt.type)

    config = ConfigParser.from_args(args, options)
    K_FOLD = config['train']['k_fold']
    OPTUNA_SEARCH = config['train']['optuna']
    MAX_MIN, MNT_METRIC = config['trainer']['kwargs']['monitor'].split()

    if OPTUNA_SEARCH:
        study = optuna.create_study(direction='maximize' if MAX_MIN == 'max' else 'minimize')
        study.optimize(objective, n_trials=10)
    else:
        main()
