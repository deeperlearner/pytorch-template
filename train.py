import importlib
import argparse
import collections

from parse_config import ConfigParser


def main(config):
    module_name = config['trainer']['module']
    class_name = config['trainer']['type']
    module_trainer = importlib.import_module(module_name, package='trainer')
    trainer_class = getattr(module_trainer, class_name)
    trainer = trainer_class(config)

    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='training')
    run_args = args.add_argument_group('run_args')
    run_args.add_argument('-c', '--config', default="config/config.json", type=str)
    run_args.add_argument('--resume', default=None, type=str)
    run_args.add_argument('--mode', default='train', type=str)
    run_args.add_argument('--run_id', default=None, type=str)
    run_args.add_argument('--log_name', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    mod_args = args.add_argument_group('mod_args')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizers;model;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loaders;train;data;args;DataLoader_args;batch_size'),
    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, default=None, type=opt.type)

    cfg = ConfigParser.from_args(args, options)
    main(cfg)
