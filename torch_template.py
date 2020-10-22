import os
import argparse
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device used:', device)

# Argument parser
parser = argparse.ArgumentParser()
## Configurations
parser.add_argument('--train_data_path', type=str, default='data')
parser.add_argument('--test_dir', type=str, default='test')
parser.add_argument('--out_dir', type=str, default='out')
args = parser.parse_args()

def main():
    return

class MyDataset(Dataset):
    def __init__(self, root_dir, mode='', transform=None):
        self.root_dir = root_dir
        filename = 'xxx'
        self.imgs_dir = os.path.join(root_dir, filename)
        self.filenames = []
        for i in range(50):
            img_file = os.path.join(root_dir, filename)
            self.filenames.append((img_file, i)) # (filename, label)

        self.transform = transform

    def __getitem__(self, idx):
        image_fn, label = self.filenames[idx]
        image = Image.open(image_fn)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filenames)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trainset = BikeDataset(root_dir, mode='train', transform=normalize)
validset = BikeDataset(root_dir, mode='valid', transform=normalize)
print("# images in trainset:", len(trainset))
print("# images in validset:", len(validset))

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# VGG16
vgg16 = models.vgg16(pretrained=True)
print(vgg16)

def train():
# Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHES):
        if (epoch+1) % 1000 == 0:
            print("Epoch:", epoch+1)
        for i, data in enumerate(trainloader):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = vgg16(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test():
    return

if __name__ == "__main__":
    main()

