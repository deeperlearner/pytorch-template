import os
import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from base import BaseDataLoader


class ImageDataset(Dataset):
    def __init__(self, data_dir, label_dir=None, mode=''):
        self.mode = mode
        # read filenames
        if self.mode == 'train':
            img_file = os.path.join(data_dir, '*.png')
            self.filenames = glob.glob(img_file)
            label_path = os.path.join(label_dir, 'train.csv')
            self.labels = pd.read_csv(label_path)
        elif self.mode == 'test':
            img_file = os.path.join(data_dir, '*.png')
            self.filenames = glob.glob(img_file)

        # transforms
        self.transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.Resize(224),
            #transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        image_file = self.filenames[idx]
        image = Image.open(image_file)

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            label = self.labels[idx]
            return image, label
        elif self.mode == 'test':
            return image, os.path.basename(image_file)

    def __len__(self):
        return len(self.filenames)

class ImageDataLoader(BaseDataLoader):
    def __init__(self, mode='train', train_dir='./data', valid_dir=None, test_dir=None, label_dir=None,
            batch_size=1, shuffle=True, validation_split=0.0, num_workers=1):
        if mode == 'train':
            # train and valid load from specified directory
            if valid_dir is not None:
                trainset = ImageDataset(train_dir, label_dir, mode)
                validset = ImageDataset(valid_dir, label_dir, mode)
                validation_split = 0.0
                super().__init__(trainset, batch_size, shuffle, validation_split, num_workers, validset)
            # train and valid with validation_split
            else:
                dataset = ImageDataset(train_dir, label_dir, mode)
                super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
        elif mode == 'test':
            assert test_dir is not None, "must specify test directory"
            testset = ImageDataset(test_dir, label_dir, mode)
            validation_split = 0.0
            super().__init__(testset, batch_size, shuffle, validation_split, num_workers)
