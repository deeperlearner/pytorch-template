from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataset(MNIST):
    def __init__(self, data_dir="./data/MNIST", mode="train"):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        super(MNISTDataset, self).__init__(
            data_dir, train=mode == "train", download=True, transform=transform
        )
        self.mode = mode
