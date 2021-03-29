import os
import logging

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils import msg_box


class BaseDataLoader(DataLoader):
    """
    Split one dataset into train data_loader and valid data_loader
    """
    # logger
    logger = logging.getLogger('data_loader')
    logger.setLevel(logging.DEBUG)

    def __init__(self, dataset, validation_split=0.0,
                 DataLoader_kwargs=None, do_transform=False):
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.split = validation_split
        self.init_kwargs = DataLoader_kwargs if DataLoader_kwargs is not None else {}

        if Cross_Valid.k_fold > 1:
            fold_msg = msg_box(f"Fold {Cross_Valid.fold_idx}")
            self.logger.info(fold_msg)
            if Cross_Valid.fold_idx == 1:
                dataset.split_cv_indexes(Cross_Valid.k_fold)
            split_idx = dataset.get_split_idx()
            train_sampler, valid_sampler = self._get_sampler(*split_idx)
            if do_transform:
                dataset.transform(split_idx)
            super().__init__(dataset, sampler=train_sampler, **self.init_kwargs)
            self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
        else:
            if validation_split > 0.0:
                split_idx = self._split_sampler()
                train_sampler, valid_sampler = self._get_sampler(*split_idx)
                if do_transform:
                    dataset.transform(split_idx)
                super().__init__(dataset, sampler=train_sampler, **self.init_kwargs)
                self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
            else:
                super().__init__(self.dataset, **self.init_kwargs)
                self.valid_loader = None

    def _get_sampler(self, train_idx, valid_idx):
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.init_kwargs['shuffle'] = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def _split_sampler(self):
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


class Cross_Valid:
    def __init__(self):
        self.log_means = pd.DataFrame()

    @classmethod
    def create_CV(cls, k_fold=1):
        cls.k_fold = k_fold
        cls.fold_idx = 1
        return cls()

    def cv_record(self, log_mean):
        # record the result of each cross validation
        self.log_means = pd.concat([self.log_means, log_mean], axis=1)
        cv_done = self.fold_idx == self.k_fold
        Cross_Valid.fold_idx += 1
        return cv_done

    def cv_result(self):
        mean = self.log_means.mean(axis=1)
        std = self.log_means.std(axis=1)
        cv_result = pd.concat([mean, std], axis=1)
        cv_result.columns = ['mean', 'std']
        return cv_result
