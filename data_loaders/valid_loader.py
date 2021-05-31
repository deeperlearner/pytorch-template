import os

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from base import BaseDataLoader
from logger import get_logger
from mains import Cross_Valid
from utils import msg_box


class ValidDataLoader(BaseDataLoader):
    """
    All cases of validation loaders
    """
    logger = get_logger('data_loader')

    def __init__(self, dataset, validation_split=0.0, DataLoader_kwargs=None,
                 do_transform=False):
        super().__init__(dataset, validation_split, DataLoader_kwargs)

        if Cross_Valid.k_fold > 1:
            fold_msg = msg_box(f"Fold {Cross_Valid.fold_idx}")
            self.logger.info(fold_msg)

        if dataset.mode in ('train', 'valid'):
            if Cross_Valid.k_fold > 1:
                split_idx = dataset.get_split_idx(Cross_Valid.fold_idx)
                train_sampler, valid_sampler = self._get_sampler(*split_idx)
            else:
                if validation_split > 0.0:
                    split_idx = self._train_valid_split()
                    train_sampler, valid_sampler = self._get_sampler(*split_idx)
                else:
                    split_idx = None
                    train_sampler, valid_sampler = None, None

            if do_transform:
                dataset.transform(split_idx)

            self.train_loader = DataLoader(dataset, sampler=train_sampler, **self.init_kwargs)
            self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
        elif dataset.mode == 'test':
            if do_transform:
                dataset.transform()
            self.test_loader = DataLoader(dataset, **self.init_kwargs)

    def _get_sampler(self, train_idx, valid_idx):
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.init_kwargs['shuffle'] = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
