import os
import argparse

import torch
import numpy as np

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import  model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


def main(args):
    config = ConfigParser(args, resume=args.model_path)
    logger = config.get_logger('train')

    # dataloader
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.valid_loader

    # model
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='training')
    args.add_argument('-c', '--config', default='config/mnist.json', type=str)
    args.add_argument('--mode', default='train', type=str)
    args.add_argument('--model_path', type=str)
    args = args.parse_args()

    main(args)
