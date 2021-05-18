import os

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from base import Cross_Valid, BaseDataLoader
from logger import get_logger
from utils import msg_box


class ImbalancedDataLoader(BaseDataLoader):
    """
    This loader will balance the ratio of class for each mini-batch
    """
    logger = get_logger('data_loader')

    def __init__(self, dataset, class_weight=None, target=None,
                 validation_split=0.0, DataLoader_kwargs=None):
        super().__init__(dataset, validation_split, DataLoader_kwargs)

        if Cross_Valid.k_fold > 1:
            fold_msg = msg_box(f"Fold {Cross_Valid.fold_idx}")
            self.logger.info(fold_msg)
            split_idx = dataset.get_split_idx(Cross_Valid.fold_idx - 1)
            train_sampler, valid_sampler = self._get_sampler(
                *split_idx, class_weight, target)
            dataset.transform(split_idx)
        else:
            if validation_split > 0.0:
                split_idx = self._train_valid_split()
                train_sampler, valid_sampler = self._get_sampler(
                    *split_idx, class_weight, target)
                dataset.transform(split_idx)
            else:
                train_sampler, valid_sampler = None, None
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **self.init_kwargs)
        self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)

    def _get_sampler(self, train_idx, valid_idx, class_weight, target):
        train_mask = np.zeros(self.n_samples)
        valid_mask = np.zeros(self.n_samples)

        train_mask[train_idx] = 1.
        valid_mask[valid_idx] = 1.

        train_weights = class_weight[target] * train_mask
        valid_weights = class_weight[target] * valid_mask

        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
        valid_sampler = WeightedRandomSampler(valid_weights, len(valid_weights))

        # turn off shuffle option which is mutually exclusive with sampler
        self.init_kwargs['shuffle'] = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
