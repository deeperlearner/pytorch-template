import os
import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, data_dir='./data', test_dir='./test', mode=''):
        self.mode = mode
        if mode == 'train':
            imgs_dir = os.path.join(data_dir, 'train')
        elif mode == 'valid':
            imgs_dir = os.path.join(data_dir, 'valid')
        elif mode == 'test':
            imgs_dir = test_dir

        # read filenames
        if mode != 'test':
            img_file = os.path.join(imgs_dir, '*.png')
            self.filenames = glob.glob(img_file)
            self.labels = []
        else:
            img_file = os.path.join(imgs_dir, '*.png')
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

class MyDataLoader(DataLoader):
    pass
