import os
import glob
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from base import BaseDataLoader


class MyDataset(Dataset):
    def __init__(self, data_dir='./data', label_path=None, mode='train'):
        # read filenames
        img_files = os.path.join(data_dir, '*.png')
        self.filenames = glob.glob(img_files)

        self.label_path = label_path
        if label_path is not None:
            self.labels = pd.read_csv(label_path, index_col=['image_name'])

        # transforms
        self.transform = transforms.Compose([
            # transforms.RandomRotation(10),
            # transforms.Resize((32,32)),
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        image_file = self.filenames[idx]
        image = Image.open(image_file).convert('RGB')

        if self.transform:
            image = self.transform(image)

        img_idx = os.path.basename(image_file)
        if self.label_path is None:
            return img_idx, image
        label = self.labels.loc[img_idx].values
        return img_idx, image, label[0]

    def __len__(self):
        return len(self.filenames)


pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
