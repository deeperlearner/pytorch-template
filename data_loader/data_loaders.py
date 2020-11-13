import os
import glob

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from base import BaseDataLoader


class ImageDataset(Dataset):
    def __init__(self, data_dir, mode=''):
        # read filenames
        if mode != 'test':
            img_file = os.path.join(data_dir, '*.png')
            self.filenames = glob.glob(img_file)
            self.labels = []
        else:
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

        if self.mode != 'test':
            label = self.labels[idx]
            return image, label
        else:
            return image, os.path.basename(image_file)

    def __len__(self):
        return len(self.filenames)

class ImageDataLoader(BaseDataLoader):
    def __init__(self, train_dir='./data', valid_dir=None, test_dir=None, batch_size=1, shuffle=True, validation_split=0.0, num_workers=1):
        if valid_dir is not None:
            trainset = ImageDataset(train_dir, mode='train')
            validset = ImageDataset(valid_dir, mode='valid')
            validation_split = 0.0
            super().__init__(trainset, batch_size, shuffle, validation_split, num_workers)
            self.valid_loader = DataLoader(validset, batch_size, shuffle, num_workers)
        else:
            dataset = ImageDataset(train_dir)
            super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

        if test_dir is not None:
            testset = ImageDataset(test_dir, mode='test')
            validation_split = 0.0
            super().__init__(testset, batch_size, shuffle, validation_split, num_workers)
