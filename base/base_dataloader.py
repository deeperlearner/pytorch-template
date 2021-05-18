import os

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base data loader
    """

    def __init__(self, dataset, validation_split=0.0, DataLoader_kwargs=None):
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.split = validation_split
        self.init_kwargs = DataLoader_kwargs if DataLoader_kwargs is not None else {}

    def _get_sampler(self, train_idx, valid_idx):
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.init_kwargs['shuffle'] = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def _train_valid_split(self):
        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(self.split, int):
            assert self.split > 0
            assert self.split < self.n_samples, \
                "validation set size is configured to be larger than entire dataset."
            len_valid = self.split
        else:
            len_valid = int(self.n_samples * self.split)

        train_idx, valid_idx = idx_full[len_valid:], idx_full[:len_valid]

        return (train_idx, valid_idx)
