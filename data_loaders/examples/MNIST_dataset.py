from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.model_selection import StratifiedKFold


class MNISTTrainset(MNIST):
    def __init__(self, data_dir='./data/MNIST'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super().__init__(data_dir, train=True, download=True, transform=transform)

    def split_cv_indexes(self, N):
        kfold = StratifiedKFold(n_splits=N)
        X, y = self.data, self.targets
        self.indexes = list(kfold.split(X, y))

    def get_split_idx(self, fold_idx):
        return self.indexes[fold_idx]


class MNISTTestset(MNIST):
    def __init__(self, data_dir='./data/MNIST'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super().__init__(data_dir, train=False, download=True, transform=transform)
