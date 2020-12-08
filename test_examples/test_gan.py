import os
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from utils import ensure_dir


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config, args):
    logger = config.get_logger('test')

    # dataloader
    testloader = config.init_obj('data_loader', module_data, mode=config.mode)

    # build model architecture
    model_G = config.init_obj('net_G', module_arch)
    model_D = config.init_obj('net_D', module_arch)
    logger.info("2-1")
    logger.info(model_G)
    logger.info(model_D)

    # get function handles of loss and metrics
    loss_fn = config.init_ftn('loss', module_loss)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    print('Loading model: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model_G = torch.nn.DataParallel(model_G)
    model_G.load_state_dict(state_dict[0])

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_G = model_G.to(device)
    model_G.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        print('testing...')
        logger.info("2-2: random generate")
        noise = (torch.randn(32, 100)).view(-1, 100, 1, 1)
        noise = noise.to(device)
        fake_image = model_G(noise)
        fake_image = fake_image.mul(0.5).add_(0.5)
        save_image(fake_image.data, args.output_path)

    logger.info('done.')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='testing')
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('--resume', type=str)
    args.add_argument('--mode', default='test', type=str)
    # additional arguments
    args.add_argument('--output_path', default='out', type=str)

    config, args = ConfigParser.from_args(args)
    main(config, args)
