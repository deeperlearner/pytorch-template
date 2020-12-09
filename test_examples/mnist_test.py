import os
import argparse
import collections

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

    sub_prob = int(args.config[-6])
    if config.mode == 'test':
        if sub_prob == 1:
            testloader = config.init_obj('data_loader', module_data, mode=config.mode)
            logger.info("3-1: lower bound")
        if sub_prob == 2:
            source_loader = config.init_obj('source_loader', module_data)
            target_loader = config.init_obj('target_loader', module_data, mode=config.mode)
            logger.info("3-2: domain adaptation")
        if sub_prob == 3:
            testloader = config.init_obj('data_loader', module_data, mode=config.mode)
            logger.info("3-3: uppper bound")

        # model
        model = config.init_obj('arch', module_arch)

        print('Loading model: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        if sub_prob == 1 or sub_prob == 3:
            with torch.no_grad():
                corrects = 0
                len_data = 0
                for i, (img_file, data, target) in enumerate(testloader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    corrects += module_metric.total_correct(output, target)
                    len_data += len(target)
                logger.info(f"accuracy: {corrects / len_data}")
        if sub_prob == 2:
            with torch.no_grad():
                corrects = 0
                len_data = 0
                for i, (img_file, data, target) in enumerate(target_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data, 0)[0]
                    corrects += module_metric.total_correct(output, target)
                    len_data += len(target)
                logger.info(f"accuracy: {corrects / len_data}")

            logger.info("3-4: t-SNE")
            with torch.no_grad():
                type_ = args.config[-9:-6]

                tsne = TSNE(n_components=2, random_state=20, verbose=1, n_iter=500, n_jobs=2)
                conv1 = model.conv1
                conv2 = model.conv2
                latent = torch.FloatTensor().to(device)
                labels = []
                for i, (img_file, data, target) in enumerate(target_loader):
                    data, target = data.to(device), target.to(device)
                    latent_output = conv2(conv1(data)).view(-1, 50*4*4)
                    latent = torch.cat((latent, latent_output.data), dim=0)
                    labels.extend(target)

                latent_sample = latent.squeeze().cpu().numpy()
                embedded_latent = tsne.fit_transform(latent_sample)

                colors_10 = np.random.rand(10,3)
                for lat, lab in zip(embedded_latent, labels):
                    plt.scatter(lat[0], lat[1], s=1, color=colors_10[lab])

                plt.savefig(f'src/fig/{type_}fig3_4a.jpg')
                plt.clf()

                tsne = TSNE(n_components=2, random_state=20, verbose=1, n_iter=500, n_jobs=2)
                latent = torch.FloatTensor().to(device)
                labels = np.zeros(0)
                for i, (img_file, data, target) in enumerate(source_loader):
                    data, target = data.to(device), target.to(device)
                    latent_output = conv2(conv1(data)).view(-1, 50*4*4)
                    latent = torch.cat((latent, latent_output.data), dim=0)
                    label = np.zeros(data.size(0), dtype=int)
                    labels = np.concatenate((labels, label))
                for i, (img_file, data, target) in enumerate(target_loader):
                    data, target = data.to(device), target.to(device)
                    latent_output = conv2(conv1(data)).view(-1, 50*4*4)
                    latent = torch.cat((latent, latent_output.data), dim=0)
                    label = np.ones(data.size(1), dtype=int)
                    labels = np.concatenate((labels, label))

                latent_sample = latent.squeeze().cpu().numpy()
                embedded_latent = tsne.fit_transform(latent_sample)

                colors_2 = np.random.rand(10,3)
                for lat, lab in zip(embedded_latent, labels):
                    lab = int(lab)
                    plt.scatter(lat[0], lat[1], s=1, color=colors_2[lab])

                plt.savefig(f'src/fig/{type_}fig3_4b.jpg')
                plt.clf()

            logger.info("3-5: model structure")
            logger.info(model)
    elif config.mode == 'inference':
        # dataloader
        testloader = config.init_obj('target_loader', module_data, mode=config.mode)
        # model
        model = config.init_obj('arch', module_arch)

        print('Loading model: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        image_id = []
        label = []
        for i, (img_file, data) in enumerate(testloader):
            data = data.to(device)
            output = model(data, 0)[0]
            image_id.extend(img_file)
            pred = output.max(1, keepdim=True)[1].squeeze()
            label.extend(pred.data.cpu().tolist())

        df = pd.DataFrame({'image_name': image_id, 'label': label})
        df.to_csv(args.output_path, index=False)

    print('done.')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='testing')
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('--resume', type=str)
    args.add_argument('--mode', default='test', type=str)
    # additional arguments
    args.add_argument('--output_path', default='hw3_data/digits/test_pred.csv', type=str)
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--test_dir'], type=str, target='target_loader;args;test_dir'),
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args)
