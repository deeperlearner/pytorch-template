from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from base import BaseDataLoader


class MnistTrainset(MNIST):
    def __init__(self, data_dir='./data/MNIST'):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super().__init__(data_dir, train=True, download=True, transform=trsfm)

class MnistTestset(MNIST):
    def __init__(self, data_dir='./data/MNIST'):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super().__init__(data_dir, train=False, download=True, transform=trsfm)
