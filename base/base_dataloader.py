import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataset:
    """
    Base class for all datasets
    """
    def __init__(self, my_dataset, data_paths: dict, mode='train'):
        self.mode = mode

        self.validset = None
        if self.mode == 'train':
            assert data_paths['train_dir'] is not None, "must specify train directory"
            # train and valid load from specified directory
            self.dataset = my_dataset(data_paths['train_dir'], label_path=data_paths['train_label'])
            if data_paths['valid_dir'] is not None:
                self.validset = my_dataset(data_paths['valid_dir'], label_path=data_paths['valid_label'])
        elif self.mode == 'test':
            assert data_paths['test_dir'] is not None, "must specify test directory"
            self.dataset = my_dataset(data_paths['test_dir'], label_path=data_paths['test_label'])
        elif self.mode == 'inference':
            self.dataset = None

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, datasets, validation_split=0.0, DataLoader_args=None):
        self.init_kwargs = DataLoader_args if DataLoader_args is not None else {}

        self.n_samples = len(datasets.dataset)

        if validation_split == 0.0:
            super().__init__(datasets.dataset, **self.init_kwargs)
            if datasets.validset is not None:
                self.valid_loader = DataLoader(datasets.validset, **self.init_kwargs)
        else:
            sampler, valid_sampler = self._split_sampler(validation_split)
            super().__init__(datasets.dataset, sampler=sampler, **self.init_kwargs)
            self.valid_loader = DataLoader(datasets.dataset, sampler=valid_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        train_idx, valid_idx = idx_full[len_valid:], idx_full[:len_valid]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.init_kwargs['shuffle'] = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
