import os

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold


class BaseDataLoader(DataLoader):
    """
    Split one dataset into train data_loader and valid data_loader
    """
    fold_idx = 0
    def __init__(self, dataset, validation_split=0.0, DataLoader_kwargs=None, N_fold=1):
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.split = validation_split
        self.init_kwargs = DataLoader_kwargs if DataLoader_kwargs is not None else {}
        self.n_fold = N_fold

        if N_fold > 1:
            if self.fold_idx == 0:
                self._cross_valid(N_fold, dataset)
            train_sampler, valid_sampler = self._split_sampler()
            super().__init__(dataset, sampler=train_sampler, **self.init_kwargs)
            self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
            self._next_fold()
        else:
            if validation_split > 0.0:
                train_sampler, valid_sampler = self._split_sampler()
                super().__init__(dataset, sampler=train_sampler, **self.init_kwargs)
                self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
            else:
                super().__init__(self.dataset, **self.init_kwargs)
                self.valid_loader = None

    def _split_sampler(self):
        idx_full = np.arange(self.n_samples)

        if self.n_fold == 1:
            np.random.seed(0)
            np.random.shuffle(idx_full)

            if isinstance(self.split, int):
                assert self.split > 0
                assert self.split < self.n_samples, "validation set size is configured to be larger than entire dataset."
                len_valid = self.split
            else:
                len_valid = int(self.n_samples * self.split)

            train_idx, valid_idx = idx_full[len_valid:], idx_full[:len_valid]
        else:
            train_idx, valid_idx = self.indexes[self.fold_idx]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.init_kwargs['shuffle'] = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    @classmethod
    def _cross_valid(cls, n_fold, dataset):
        kfold = KFold(n_splits = n_fold)
        cls.indexes = [(train_idx, valid_idx) for train_idx, valid_idx in kfold.split(dataset)]

    @classmethod
    def _next_fold(cls):
        cls.fold_idx += 1
