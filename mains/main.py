import os
import sys
import argparse
import collections
import time

import optuna

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from logger import get_logger
from parse_config import ConfigParser
from utils import msg_box, consuming_time


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='training')
    run_args = args.add_argument_group('run_args')
    run_args.add_argument('-c', '--config', default="configs/config.json", type=str)
    run_args.add_argument('--resume', default=None, type=str)
    run_args.add_argument('--mode', default='train', type=str)
    run_args.add_argument('--run_id', default=None, type=str)
    run_args.add_argument('--log_name', default=None, type=str)
    run_args.add_argument('--mp', action='store_true', help="multiprocessing")

    # custom cli options to modify configuration from default values given in json file.
    mod_args = args.add_argument_group('mod_args')
    CustomArgs = collections.namedtuple('CustomArgs', "flags type target")
    options = [
        CustomArgs(['--name'], type=str, target="name"),
        CustomArgs(['--num_workers'], type=int, target="data_loaders;train;data;kwargs;DataLoader_kwargs;num_workers"),
        CustomArgs(['--lr', '--learning_rate'], type=float, target="optimizers;model;args;lr"),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target="data_loaders;train;data;args;DataLoader_kwargs;batch_size"),
    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, type=opt.type)

    config = ConfigParser.from_args(args, options)
    logger = get_logger('main')
    msg = msg_box("TRAIN")
    logger.debug(msg)
    max_min, mnt_metric = config['trainer']['kwargs']['monitor'].split()

    n_trials = config['optuna']['n_trials']
    if n_trials > 0:
        objective = config.init_obj(['optuna'])

        direction = 'maximize' if max_min == 'max' else 'minimize'
        start = time.time()
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)

        msg = msg_box("Optuna result")
        end = time.time()
        total_time = consuming_time(start, end)
        msg += f"\nConsuming time: {total_time}."
        msg += f"\nM{direction[1:-3]}al {mnt_metric}: {study.best_value:.6f}"
        msg += f"\nBest hyperparameters: {study.best_params}"
        logger.info(msg)
    else:
        train(config)
