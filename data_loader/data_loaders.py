import os
import glob

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image

from parse_config import ConfigParser

class ImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        self.transform = normalize
        self.filenames = []

        # read filenames
        filenames = glob.glob(os.path.join(data_dir, '*.png'))
        for fn in filenames:
            self.filenames.append((fn, "label")) # (filename, label)

        self.transform = transform

    def __getitem__(self, idx):
        image_file, label = self.filenames[idx]
        image = Image.open(image_file)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filenames)

