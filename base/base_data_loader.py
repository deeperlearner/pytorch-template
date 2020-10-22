import os
import glob
from PIL import Image

import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader


class BaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.filenames = []
        filenames = glob.glob(os.path.join(data_dir, '*.png'))
        for fn in filenames:
            self.filenames.append((fn, "label")) # (filename, label)

        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.filenames[idx]
        image = Image.open(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filenames)

