from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from base import BaseDataLoader


class MNISTTrainset(MNIST):
    def __init__(self, data_dir='./data/MNIST'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super().__init__(data_dir, train=True, download=True, transform=transform)


class MNISTTestset(MNIST):
    def __init__(self, data_dir='./data/MNIST'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super().__init__(data_dir, train=False, download=True, transform=transform)
