import os

import numpy as np
from torch.utils.data import DataLoader

from base import Cross_Valid, BaseDataLoader
from logger import get_logger
from utils import msg_box


class ValidDataLoader(BaseDataLoader):
    """
    All cases of validation loaders
    """
    logger = get_logger('data_loader')

    def __init__(self, dataset, validation_split=0.0, DataLoader_kwargs=None):
        super().__init__(dataset, validation_split, DataLoader_kwargs)

        if Cross_Valid.k_fold > 1:
            fold_msg = msg_box(f"Fold {Cross_Valid.fold_idx}")
            self.logger.info(fold_msg)
            split_idx = dataset.get_split_idx(Cross_Valid.fold_idx - 1)
            train_sampler, valid_sampler = self._get_sampler(*split_idx)
        else:
            if validation_split > 0.0:
                split_idx = self._train_valid_split()
                train_sampler, valid_sampler = self._get_sampler(*split_idx)
            else:
                train_sampler, valid_sampler = None, None
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **self.init_kwargs)
        self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
