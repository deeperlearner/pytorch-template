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
    model = config.init_obj('arch', module_arch, mode='test')
    logger.info("1-1")
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = config.init_ftn('loss', module_loss)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

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

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        print('testing...')
        if config.mode == 'test':
            logger.info("1-3: VAE on test")
            for i, (data, target) in enumerate(testloader):
                data = data.to(device)
                output = model(data)[0]
                output = output.mul(0.5).add_(0.5)
                origin = data.mul(0.5).add_(0.5)

                dir_path = os.path.dirname(args.output_path)
                out_path = os.path.join(dir_path, 'Table1_3')
                ensure_dir(out_path)
                MSEs = []
                for j in range(10):
                    save_file = os.path.join(out_path, target[j])
                    save_image(origin[j], save_file)
                    save_file = os.path.join(out_path, 'Recon_' + target[j])
                    save_image(output[j], save_file)
                    loss_fn = nn.MSELoss()
                    MSEs.append(loss_fn(output[j], origin[j]).item())
                logger.info("MSEs:")
                logger.info(MSEs)
                break

        logger.info("1-4: random generate")
        decoder = model.decoder
        noise = torch.randn(32, 1024).view(-1, 1024, 1, 1)
        noise = noise.to(device)
        predict = decoder(noise)
        predict = predict.mul(0.5).add_(0.5)

        if config.mode == 'test':
            dir_name = os.path.dirname(args.output_path)
            ensure_dir(dir_name)
        save_image(predict, args.output_path, nrow=8)

        if config.mode == 'test':
            logger.info("1-5: TSNE")
            tsne = TSNE(n_components=2, random_state=20, verbose=1, n_iter=1000)
            encoder = model.encoder
            mu = model.mu
            latent = torch.FloatTensor().to(device)
            test_filenames = []
            for i, (data, target) in enumerate(testloader):
                data = data.to(device)
                encoded = encoder(data)
                latent_output = mu(encoded)
                latent = torch.cat((latent, latent_output.data), dim=0)
                test_filenames.extend(target)

            latent_sample = latent.squeeze().cpu().numpy()
            embedded_latent = tsne.fit_transform(latent_sample)

            gender = pd.read_csv(os.path.join("hw3_data/face", 'test.csv'), index_col=['image_name'])
            gender = gender['Male']
            m, f = 0, 0
            for lat, file_ in zip(embedded_latent, test_filenames):
                gen = gender[file_]
                if gen == 1.0:
                    plt.scatter(lat[0], lat[1], c='b', alpha=0.3, label='Male' if not m else '')
                    m = 1
                else:
                    plt.scatter(lat[0], lat[1], c='r', alpha=0.3, label='Female' if not f else '')
                    f = 1

            plt.legend(loc="best")
            dir_path = os.path.dirname(args.output_path)
            save_path = os.path.join(dir_path, 'fig1_5.jpg')
            plt.savefig(save_path)

    logger.info('done.')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='testing')
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('--resume', type=str)
    args.add_argument('--mode', default='test', type=str)
    # additional arguments
    args.add_argument('--output_path', type=str)

    config, args = ConfigParser.from_args(args)
    main(config, args)