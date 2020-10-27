import os
import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_dir, mode=''):
        if mode == 'train':
            train_valid = 'train_50'
            num_j = 450
        elif mode == 'valid':
            train_valid = 'val_50'
            num_j = 50

        imgs_dir = os.path.join(root_dir, train_valid)
        self.filenames = []
        # read filenames
        for i in range(50):
            for j in range(num_j):
                img_file = os.path.join(imgs_dir, f"{i}_{j}.png")
                self.filenames.append((img_file, i)) # (filenames, label)

        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        image_file, label = self.filenames[idx]
        image = Image.open(image_file)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filenames)

class MyDataLoader(DataLoader):
    pass
