import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import argparse
import collections

import torch
import torch.nn as nn
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from logger import get_logger
from mains import Cross_Valid
import models.loss as module_loss
import models.metric as module_metric
from models.metric import MetricTracker
from parse_config import ConfigParser
from utils import ensure_dir, prepare_device, get_by_path, msg_box

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main():
    logger = get_logger('test')
    test_msg = msg_box("TEST")
    logger.debug(test_msg)

    # setup GPU device if available, move model into configured device
    device, device_ids = prepare_device(config['n_gpu'])

    # datasets
    test_datasets = dict()
    keys = ['datasets', 'test']
    for name in get_by_path(config, keys):
        test_datasets[name] = config.init_obj([*keys, name], 'data_loaders')
    ## compute inverse class frequency as class weight
    if config['datasets'].get('imbalanced', False):
        target = test_datasets['data'].y_test  # TODO
        class_weight = compute_class_weight(class_weight='balanced',
                                            classes=target.unique(),
                                            y=target)
        class_weight = torch.FloatTensor(class_weight).to(device)

    results = pd.DataFrame()
    Cross_Valid.create_CV(K_FOLD)
    for k in range(K_FOLD):
        # data_loaders
        test_data_loaders = dict()
        keys = ['data_loaders', 'test']
        for name in get_by_path(config, keys):
            dataset = test_datasets[name]
            loaders = config.init_obj([*keys, name], 'data_loaders', dataset)
            test_data_loaders[name] = loaders.test_loader

        # models
        if K_FOLD > 1:
            fold_prefix = f'fold_{k}_'
            dirname = os.path.dirname(config.resume)
            basename = os.path.basename(config.resume)
            resume = os.path.join(dirname, fold_prefix + basename)
        else:
            resume = config.resume
        logger.info(f"Loading model: {resume} ...")
        checkpoint = torch.load(resume)
        models = dict()
        logger_model = get_logger('model', verbosity=0)
        for name in config['models']:
            model = config.init_obj(['models', name], 'models')
            logger_model.info(model)
            state_dict = checkpoint['models'][name]
            if config['n_gpu'] > 1:
                model = torch.nn.DataParallel(model)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            models[name] = model
        model = models['model']

        # losses
        kwargs = {}
        if 'balanced' in get_by_path(config, ['losses', 'loss', 'type']):
            kwargs.update(class_weight=class_weight)
        loss_fn = config.init_obj(['losses', 'loss'], module_loss, **kwargs)

        # metrics
        metrics_iter = [getattr(module_metric, met) for met in config['metrics']['per_iteration']]
        metrics_epoch = [getattr(module_metric, met) for met in config['metrics']['per_epoch']]
        keys_loss = ['loss']
        keys_iter = [m.__name__ for m in metrics_iter]
        keys_epoch = [m.__name__ for m in metrics_epoch]
        test_metrics = MetricTracker(keys_loss + keys_iter, keys_epoch)

        with torch.no_grad():
            print("testing...")
            test_loader = test_data_loaders['data']

            if len(metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(device)
                targets = torch.FloatTensor().to(device)
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                if len(metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output))
                    targets = torch.cat((targets, target))

                #
                # save sample images, or do something with output here
                #

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                test_metrics.iter_update('loss', loss.item())
                for met in metrics_iter:
                    test_metrics.iter_update(met.__name__, met(target, output))

            for met in metrics_epoch:
                test_metrics.epoch_update(met.__name__, met(targets, outputs))

        test_log = test_metrics.result()
        results = pd.concat((results, test_log))
        logger.info(test_log)

        # cross validation
        if K_FOLD > 1:
            Cross_Valid.next_fold()

    # result
    msg = msg_box("result")
    sum_result = results.groupby(results.index)
    avg_result = sum_result.mean(numeric_only=False)
    logger.info(f"{msg}\n{avg_result}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='testing')
    run_args = args.add_argument_group('run_args')
    run_args.add_argument('-c', '--config', default="configs/examples/mnist.json", type=str)
    run_args.add_argument('--resume', default=None, type=str)
    run_args.add_argument('--mode', default='test', type=str)
    run_args.add_argument('--run_id', default=None, type=str)
    run_args.add_argument('--log_name', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    mod_args = args.add_argument_group('mod_args')
    CustomArgs = collections.namedtuple('CustomArgs', "flags type target")
    options = [
    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, type=opt.type)

    # config.test_args: additional arguments for testing
    test_args = args.add_argument_group('test_args')
    test_args.add_argument('--output_path', default=None, type=str)

    config = ConfigParser.from_args(args, options)
    K_FOLD = config['train']['k_fold']

    main()
