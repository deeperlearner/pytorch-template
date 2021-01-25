import os
import argparse
import collections

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from parse_config import ConfigParser
import data_loader as module_data
import model as module_arch
import model.loss as module_loss
import model.metric as module_metric
from utils import ensure_dir


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    # test_args: config.test_args

    logger = config.get_logger('test')

    # dataset
    testset = config.init_obj('datasets', 'data', module_data, mode=config.mode)

    # data_loader
    testloader = config.init_obj('data_loaders', 'data', module_data, testset)

    # prepare model for testing
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # models
    logger.info('Loading model: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    models = dict()
    for name in config['models']:
        # init model
        model = config.init_obj('models', name, module_arch)
        logger.info(model)
        # load model state_dict
        state_dict = checkpoint['models'][name]
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        # to gpu & eval mode
        model = model.to(device)
        model.eval()
        # store to models
        models[name] = model

    # get function handles of loss and metrics
    loss_fn = config.init_ftn('losses', 'loss', module_loss)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        print('testing...')
        for batch_idx, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(testloader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='testing')
    run_args = args.add_argument_group('run_args')
    run_args.add_argument('-c', '--config', default="config/examples/mnist.json", type=str)
    run_args.add_argument('--resume', default=None, type=str)
    run_args.add_argument('--mode', default='test', type=str)
    run_args.add_argument('--log_name', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    mod_args = args.add_argument_group('mod_args')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags default type target')
    options = [
        #CustomArgs(['--bs', '--batch_size'], default=1, type=int, target='data_loaders;data;args;DataLoader_args;batch_size'),
        #CustomArgs(['--shuffle'], default=False, type=bool, target='data_loaders;data;args;DataLoader_args;shuffle'),
    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, default=opt.default, type=opt.type)

    # additional arguments for testing
    test_args = args.add_argument_group('test_args')
    test_args.add_argument('--test_dir', default=None, type=str)
    test_args.add_argument('--out_dir', default=None, type=str)
    test_args.add_argument('--output_path', default=None, type=str)

    cfg = ConfigParser.from_args(args, options)
    main(cfg)
