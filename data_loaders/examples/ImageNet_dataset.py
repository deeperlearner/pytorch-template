import os
import glob
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageNetDataset(Dataset):
    def __init__(self, data_dir="./data/ImageNet", mode="train"):
        self.mode = mode
        if mode == "train":
            imgs_dir = os.path.join(data_dir, "train_50")
            num_j = range(450)
        elif mode == "valid":
            imgs_dir = os.path.join(data_dir, "val_50")
            num_j = range(450, 500)
        elif mode == "test":
            imgs_dir = data_dir

        # read filenames
        if mode in ("train", "valid"):
            self.filenames = []
            self.labels = []
            for i in range(50):
                for j in num_j:
                    img_file = os.path.join(imgs_dir, f"{i}_{j}.png")
                    self.filenames.append(img_file)
                    self.labels.append(i)
        else:  # test
            img_file = os.path.join(imgs_dir, "*.png")
            self.filenames = glob.glob(img_file)

        # Normalization
        self.transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.Resize(224),
                # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx):
        image_file = self.filenames[idx]
        if self.mode != "test":
            label = self.labels[idx]

        image = Image.open(image_file)

        if self.transform:
            image = self.transform(image)

        if self.mode != "test":
            return image, label
        else:
            return image, os.path.basename(image_file)

    def __len__(self):
        return len(self.filenames)


pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)
