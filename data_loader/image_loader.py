import os
import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from base import BaseDataLoader


class ImageDataset(Dataset):
    def __init__(self, data_dir, label_path=None):
        # read filenames
        img_files = os.path.join(data_dir, '*.png')
        self.filenames = glob.glob(img_files)

        self.label_path = label_path
        if label_path is not None:
            self.labels = pd.read_csv(label_path, index_col=['image_name'])

        # transforms
        self.transform = transforms.Compose([
            #transforms.RandomRotation(10),
            #transforms.Resize(28),
            #transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                  std=(0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        image_file = self.filenames[idx]
        image = Image.open(image_file).convert('RGB')

        if self.transform:
            image = self.transform(image)

        img_idx = os.path.basename(image_file)
        if self.label_path is not None:
            label = self.labels.loc[img_idx].values
            return img_idx, image, label[0]
        else:
            return img_idx, image

    def __len__(self):
        return len(self.filenames)

class ImageDataLoader(BaseDataLoader):
    def __init__(self, **kwargs):
        super().__init__(ImageDataset, **kwargs)
