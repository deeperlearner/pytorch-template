from torchvision import transforms
from torchvision.datasets import MNIST

from base import BaseDataLoader


class MnistDataset(MNIST):
    def __init__(self, data_dir='./data', label_path=None):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super().__init__(data_dir, download=True, transform=trsfm)
