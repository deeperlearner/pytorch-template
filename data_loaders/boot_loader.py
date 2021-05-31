import os

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from base import BaseDataLoader
from logger import get_logger
from mains import Cross_Valid
from utils import msg_box


class BootstrapDataLoader(BaseDataLoader):
    """
    This loader performs bootstrap statistics but not for booting your PC ;)
    """
    logger = get_logger('data_loader')

    def __init__(self, dataset, number=0, sample_size=None,
                 DataLoader_kwargs=None, do_transform=False):
        super().__init__(dataset, 0.0, DataLoader_kwargs)

        assert dataset.mode == 'test', "bootstrap method only support for testing."
        number_msg = msg_box(f"Booststrap No. {number}")
        self.logger.info(number_msg)

        if do_transform:
            dataset.transform()

        N = self.n_samples if sample_size is None else sample_size
        boot_sampler = RandomSampler(dataset, replacement=True, num_samples=N)
        assert not self.init_kwargs['shuffle'], "shuffle need to be set False"
        self.test_loader = DataLoader(dataset, sampler=boot_sampler, **self.init_kwargs)
