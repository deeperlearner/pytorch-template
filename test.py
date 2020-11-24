import os
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
report = True

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.models as module_arch
from parse_config import ConfigParser
from utils import ensure_dir


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args):
    config = ConfigParser(args)
    logger = config.get_logger('test')

    # dataloader
    testloader = config.init_obj('data_loader', module_data, mode=args.mode)

    # build model architecture
    model = config.init_obj('arch', module_arch, mode='test')
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

        df = pd.DataFrame({'image_id': image_id, 'label': label})
        out_csv = os.path.join(args.out_dir, 'test_pred.csv')
        df.to_csv(out_csv, index=False)

        if report:
            print("1-3: VAE on test")
            for i, (data, target) in enumerate(testloader):
                data = data.to(device)
                output = model(data)[0]
                output = output.mul(0.5).add_(0.5)
                origin = data.mul(0.5).add_(0.5)

                out_path = 'p1/fig/Table_3'
                ensure_dir(out_path)
                MSEs = []
                for j in range(10):
                    save_file = os.path.join(out_path, target[j])
                    save_image(origin[j], save_file)
                    save_file = os.path.join(out_path, 'Recon_' + target[j])
                    save_image(output[j], save_file)
                    loss_fn = nn.MSELoss()
                    MSEs.append(loss_fn(output[j], origin[j]).item())
                print("MSE:", MSEs)
                break

        print("1-4: random generate")
        decoder = model.decoder
        noise = torch.randn(32, 1024).view(-1, 1024, 1, 1)
        noise = noise.to(device)
        predict = decoder(noise)
        predict = predict.mul(0.5).add_(0.5)

        save_image(predict, args.output_path, nrow=8)

        if report:
            print("1-5: TSNE")
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
                    plt.scatter(lat[0], lat[1], c='b', label='Male' if not m else '')
                    m = 1
                else:
                    plt.scatter(lat[0], lat[1], c='r', label='Female' if not f else '')
                    f = 1

            plt.legend(loc="best")
            save_path = os.path.join('p1/fig', 'fig1_5.jpg')
            plt.savefig(save_path)

    logger.debug('done.')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='testing')
    args.add_argument('-c', '--config', default='config/mnist.json', type=str)
    args.add_argument('--model_path', type=str)
    args.add_argument('--mode', default='test', type=str)
    args.add_argument('--test_dir', default='test', type=str)
    args.add_argument('--out_dir', default='out', type=str)
    args.add_argument('--output_path', default='out', type=str)
    args = args.parse_args()

    main(args)
