from torchvision import transforms
from torchvision.datasets import MNIST

from base import BaseDataLoader


class MnistDataset(MNIST):
    def __init__(self, data_dir):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super().__init__(data_dir, download=True, transform=trsfm)

class MnistDataLoader(BaseDataLoader):
    def __init__(self, train_dir='./data', valid_dir=None, test_dir=None, batch_size=1, shuffle=True, validation_split=0.1, num_workers=1):
        dataset = MnistDataset(train_dir)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
