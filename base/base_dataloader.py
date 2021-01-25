import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Split one dataset into train data_loader and valid data_loader
    """
    def __init__(self, dataset, N_fold=1, validation_split=0.0, DataLoader_kwargs=None):
        self.init_kwargs = DataLoader_kwargs if DataLoader_kwargs is not None else {}

        self.n_samples = len(dataset)

        if N_fold > 1:
            pass
        else:
            if validation_split > 0.0:
                sampler, valid_sampler = self._split_sampler(validation_split)
                super().__init__(dataset, sampler=sampler, **self.init_kwargs)
                self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
            else:
                super().__init__(dataset, **self.init_kwargs)
                self.valid_loader = None

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
