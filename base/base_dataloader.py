import os

import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from logger import get_logger
from mains import Cross_Valid
from utils import msg_box


class BaseDataLoader(DataLoader):
    """
    Base data loader
    """

    logger = get_logger("data_loader")

    def __init__(self, dataset, validation_split=0.0, DataLoader_kwargs=None):
        self.n_samples = len(dataset)
        self.split = validation_split
        self.init_kwargs = DataLoader_kwargs if DataLoader_kwargs is not None else {}

        if Cross_Valid.k_fold > 1:
            fold_msg = msg_box(f"Fold {Cross_Valid.fold_idx}")
            self.logger.info(fold_msg)

    def _train_valid_split(self, labels=None):
        samples_array = np.arange(self.n_samples)
        SEED = Cross_Valid.repeat_idx
        return train_test_split(
            samples_array,
            test_size=self.split,
            random_state=SEED,
            stratify=labels,
        )
