import os
import argparse
import collections

import torch

from parse_config import ConfigParser
import trainer as module_trainer


def main(config):
    trainer = config.init_obj(None, 'trainer', module_trainer, config)

    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='training')
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('--resume', default=None, type=str)
    args.add_argument('--mode', default='train', type=str)
    args.add_argument('--run_id', default=None, type=str)
    args.add_argument('--log_name', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--lm'], type=float, target='loss;args;lm'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
