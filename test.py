import os
import argparse

import torch
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
plotting = False

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import ensure_dir


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args):
    config = ConfigParser(args, resume=args.model_path)
    logger = config.get_logger('test')

    # dataloader
    testloader = config.init_obj('data_loader', module_data, testset)

    # build model architecture
    model = config.init_obj('model', module_arch)
    logger.debug(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.debug('Loading model: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    image_id = []
    label = []
    with torch.no_grad():
        logger.debug('testing...')
        for i, (data, target) in enumerate(testloader):
            data = data.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1].squeeze()
            #
            # save sample images, or do something with output here
            #
            image_id.extend(target)
            label.extend(pred.data.cpu().tolist())
            output_path = os.path.join(args.output_dir, target)

            ## computing loss, metrics on test set
            #loss = loss_fn(output, target)
            #batch_size = data.shape[0]
            #total_loss += loss.item() * batch_size
            #for i, metric in enumerate(metric_fns):
            #    total_metrics[i] += metric(output, target) * batch_size

    df = pd.DataFrame({'image_id': image_id, 'label': label})
    out_csv = os.path.join(args.out_dir, 'test_pred.csv')
    df.to_csv(out_csv, index=False)
    logger.debug('done.')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='testing')
    args.add_argument('-c', '--config', default='config/mnist.json', type=str)
    args.add_argument('--mode', default='test', type=str)
    args.add_argument('--model_path', type=str)
    args.add_argument('--test_dir', default='test', type=str)
    args.add_argument('--out_dir', default='out', type=str)
    args = args.parse_args()

    main(args)
