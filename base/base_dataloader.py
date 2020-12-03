import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, MyDataset, 
            batch_size=1, shuffle=False, num_workers=1,
            mode='train', validation_split=0.0,
            train_dir=None, valid_dir=None, test_dir=None,
            train_label_path=None, valid_label_path=None, test_label_path=None):

        if mode == 'train':
            assert train_dir is not None, "must specify train directory"
            # train and valid load from specified directory
            dataset = MyDataset(train_dir, label_path=train_label_path)
            validset = None
            if valid_dir is not None:
                validset = MyDataset(valid_dir, label_path=valid_label_path)
        elif mode == 'test':
            assert test_dir is not None, "must specify test directory"
            dataset = MyDataset(test_dir, label_path=test_label_path)
        elif mode == 'inference':
            return

        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': default_collate,
            'num_workers': num_workers
        }
        if validation_split == 0.0:
            super().__init__(**self.init_kwargs)
            self.valid_loader = None
            if validset is not None:
                self.valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=num_workers)
        else:
            sampler, valid_sampler = self._split_sampler(validation_split)
            super().__init__(sampler=sampler, **self.init_kwargs)
            self.valid_loader = DataLoader(sampler=valid_sampler, **self.init_kwargs)

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
