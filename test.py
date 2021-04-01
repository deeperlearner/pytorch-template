import os
import argparse
import collections
import time

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from base import Cross_Valid
from logger import get_logger
import model.loss as module_loss
import model.metric as module_metric
from model.metric import MetricTracker, AUROC, AUPRC
from parse_config import ConfigParser
from utils import ensure_dir, prepare_device, get_by_path, msg_box

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # starting time
    start = time.time()

    k_fold = config['trainer'].get('k_fold', 1)
    fold_idx = config['trainer'].get('fold_idx', 0)

    logger = get_logger('test')
    test_msg = msg_box("TEST")
    logger.debug(test_msg)

    # datasets
    test_datasets = dict()
    keys = ['datasets', 'test']
    for name in get_by_path(config, keys):
        test_datasets[name] = config.init_obj([*keys, name], 'data_loader')

    # data_loaders
    test_data_loaders = dict()
    keys = ['data_loaders', 'test']
    for name in get_by_path(config, keys):
        dataset = test_datasets[name]
        do_transform = get_by_path(config, [*keys, name]).get('do_transform', False)
        if do_transform:
            dataset.transform()
        test_data_loaders[name] = config.init_obj([*keys, name], 'data_loader', dataset)

    # prepare model for testing
    device, device_ids = prepare_device(config['n_gpu'])

    CV_manager = Cross_Valid.create_CV(k_fold=k_fold)
    for fold_idx in range(1, k_fold + 1):
        # models
        if k_fold > 1:
            fold_prefix = f'fold_{fold_idx}_'
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
            model = config.init_obj(['models', name], 'model')
            logger_model.info(model)
            state_dict = checkpoint['models'][name]
            if config['n_gpu'] > 1:
                model = torch.nn.DataParallel(model)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            models[name] = model

        # loss
        loss_fn = config.init_ftn(['losses', 'loss'], module_loss)

        # metrics
        metrics_iter = [getattr(module_metric, met) for met in config['metrics']['per_iteration']]
        metrics_epoch = [getattr(module_metric, met) for met in config['metrics']['per_epoch']]
        keys_loss = ['loss']
        keys_iter = [m.__name__ for m in metrics_iter]
        keys_epoch = [m.__name__ for m in metrics_epoch]
        test_metrics = MetricTracker(keys_loss + keys_iter, keys_epoch)

        with torch.no_grad():
            print("testing...")
            model = models['model']
            testloader = test_data_loaders['data']
            if len(metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(device)
                targets = torch.FloatTensor().to(device)
            for batch_idx, (data, target) in enumerate(testloader):
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
                    test_metrics.iter_update(met.__name__, met(output, target))

            for met in metrics_epoch:
                test_metrics.epoch_update(met.__name__, met(outputs, targets))

        test_log = test_metrics.result()
        logger.info(test_log)
        # cross validation is enabled
        if k_fold > 1:
            log_mean = test_log['mean']
            if CV_manager.cv_record(log_mean):
                # done and print result
                cv_result = CV_manager.cv_result()
                end = time.time()
                ty_res = time.gmtime(end - start)
                res = time.strftime("%H hours, %M minutes, %S seconds", ty_res)
                k_fold_msg = msg_box(f"{k_fold}-fold cross validation ensembled result")
                logger.info(f"{k_fold_msg}\n"
                            f"Total running time: {res}\n"
                            f"{cv_result}\n")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='testing')
    run_args = args.add_argument_group('run_args')
    run_args.add_argument('-c', '--config', default="config/examples/mnist.json", type=str)
    run_args.add_argument('--resume', default=None, type=str)
    run_args.add_argument('--mode', default='test', type=str)
    run_args.add_argument('--run_id', default=None, type=str)
    run_args.add_argument('--log_name', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    mod_args = args.add_argument_group('mod_args')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags default type target')
    options = [
        # CustomArgs(['--bs', '--batch_size'], default=1, type=int,
        #            target='data_loaders;data;args;DataLoader_args;batch_size'),
        # CustomArgs(['--shuffle'], default=False, type=bool, target='data_loaders;data;args;DataLoader_args;shuffle'),
    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, default=opt.default, type=opt.type)

    # additional arguments for testing
    test_args = args.add_argument_group('test_args')
    test_args.add_argument('--output_path', default=None, type=str)

    cfg = ConfigParser.from_args(args, options)
    main(cfg)
