import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        if validation_split == 0.0:
            super().__init__(**self.init_kwargs)
            self.valid_loader = None
        else:
            self.sampler, self.valid_sampler = self._split_sampler(validation_split)
            super().__init__(sampler=self.sampler, **self.init_kwargs)
            self.valid_loader = self.split_validation()

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

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
