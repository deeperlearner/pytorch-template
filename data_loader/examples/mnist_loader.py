from torchvision import transforms
from torchvision.datasets import MNIST

from base import BaseDataset, BaseDataLoader


class MnistDataset(MNIST):
    def __init__(self, data_dir='./data', label_path=None, mode='train'):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super().__init__(data_dir, train=MnistDataset.train, download=True, transform=trsfm)

class MyDataset(BaseDataset):
    def __init__(self, data_paths: dict, mode='train'):
        MnistDataset.train = mode == 'train'
        super().__init__(MnistDataset, data_paths, mode)

class MyDataLoader(BaseDataLoader):
    pass
