import os

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold


class BaseDataLoader(DataLoader):
    """
    Split one dataset into train data_loader and valid data_loader
    """
    N_fold = 1
    fold_idx = 1
    log_means = pd.DataFrame()

    def __init__(self, dataset, validation_split=0.0, DataLoader_kwargs=None, normalize=False):
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.split = validation_split
        self.init_kwargs = DataLoader_kwargs if DataLoader_kwargs is not None else {}

        if self.N_fold > 1:
            if self.fold_idx == 1:
                self.split_cv_indexes(dataset)
            train_sampler, valid_sampler, split_idx = self._split_sampler()
            if normalize:
                dataset.normalize(split_idx)
            super().__init__(dataset, sampler=train_sampler, **self.init_kwargs)
            self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
        else:
            if validation_split > 0.0:
                train_sampler, valid_sampler, split_idx = self._split_sampler()
                if normalize:
                    dataset.normalize(split_idx)
                super().__init__(dataset, sampler=train_sampler, **self.init_kwargs)
                self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
            else:
                super().__init__(self.dataset, **self.init_kwargs)
                self.valid_loader = None

    def _split_sampler(self):
        idx_full = np.arange(self.n_samples)

        if self.N_fold == 1:
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
        else:
            train_idx, valid_idx = next(self.indexes)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.init_kwargs['shuffle'] = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler, (train_idx, valid_idx)

    @classmethod
    def split_cv_indexes(cls, dataset):
        kfold = KFold(n_splits=cls.N_fold)
        cls.indexes = kfold.split(dataset)

    @classmethod
    def cv_record(cls, log_mean):
        # record the result of each cross validation
        cls.log_means = pd.concat([cls.log_means, log_mean], axis=1)
        cv_done = cls.fold_idx == cls.N_fold
        cls.fold_idx += 1
        return cv_done

    @classmethod
    def cv_result(cls):
        mean = cls.log_means.mean(axis=1)
        std = cls.log_means.std(axis=1)
        cv_result = pd.concat([mean, std], axis=1)
        cv_result.columns = ['mean', 'std']
        return cv_result
