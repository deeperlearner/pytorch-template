import os

import requests
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from base import BaseDataLoader


class AdultDataset(Dataset):

    resources = ["http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                 "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
                 "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]

    def __init__(self, data_dir='./data/Adult', mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.download(data_dir)

        columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                   "marital-status", "occupation", "relationship", "race", "sex",
                   "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        train_path = os.path.join(data_dir, 'adult.data')
        train_data = pd.read_csv(train_path, sep=', ', names=columns, engine='python', na_values='?')
        len_data = len(train_data.index)

        test_path = os.path.join(data_dir, 'adult.test')
        test_data = pd.read_csv(test_path, sep=', ', names=columns, engine='python', skiprows=1, na_values='?')
        len_test = len(test_data.index)

        merged_data = pd.concat([train_data, test_data])
        self.data = merged_data.drop(["income"], axis=1)
        self.label = merged_data.loc[:, "income"]

        num_data = self.data.select_dtypes(include=['int'])
        cat_data = self.data.select_dtypes(include=['object'])
        cat_data = pd.get_dummies(cat_data)
        self._preprocess()
        # adult.data
        self.x_num_data = num_data.iloc[:len_data]
        self.x_cat_data = cat_data.iloc[:len_data]
        self.y_data = self.label.iloc[:len_data]
        # adult.test
        self.x_num_test = num_data.iloc[len_data:]
        self.x_cat_test = cat_data.iloc[len_data:]
        self.y_test = self.label.iloc[len_data:]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        if self.mode == 'train':
            return len(self.y_data)
        return len(self.y_test)

    def _preprocess(self):
        self.label.replace(to_replace=r'<=50K.?', value=0, regex=True, inplace=True)
        self.label.replace(to_replace=r'>50K.?', value=1, regex=True, inplace=True)

    def split_cv_indexes(self, N):
        kfold = StratifiedKFold(n_splits=N)
        X, y = self.x_num_data, self.y_data
        self.indexes = kfold.split(X, y)

    def get_split_idx(self):
        return next(self.indexes)

    def transform(self, split_idx=None):
        self.compute_data_info(split_idx)
        self.impute()
        self.normalize()

    def compute_data_info(self, split_idx):
        if self.mode == 'train':
            # compute mean, std, mode on training data only!
            train_idx, valid_idx = split_idx
            self.num_mean = self.x_num_data.iloc[train_idx].mean()
            self.num_std = self.x_num_data.iloc[train_idx].std()
            self.cat_mode = self.x_cat_data.iloc[train_idx].mode().loc[0]
        elif self.mode == 'test':
            self.num_mean = self.x_num_data.mean()
            self.num_std = self.x_num_data.std()
            self.cat_mode = self.x_cat_data.mode().loc[0]

    def impute(self):
        # Simple impute
        if self.mode == 'train':
            self.x_num_data = self.x_num_data.fillna(self.num_mean)
            self.x_cat_data = self.x_cat_data.fillna(self.cat_mode)
        elif self.mode == 'test':
            self.x_num_test = self.x_num_test.fillna(self.num_mean)
            self.x_cat_test = self.x_cat_test.fillna(self.cat_mode)

    def normalize(self):
        if self.mode == 'train':
            x_num, x_cat = self.x_num_data, self.x_cat_data
            y = self.y_data
        elif self.mode == 'test':
            x_num, x_cat = self.x_num_test, self.x_cat_test
            y = self.y_test

        # normalize on numerical data only!
        x_num = (x_num - self.num_mean) / self.num_std
        # DataFrame to Tensor
        x_num = torch.from_numpy(x_num.values.astype(np.float32))
        x_cat = torch.from_numpy(x_cat.values.astype(np.float32))
        y = torch.from_numpy(y.to_numpy().astype(np.float32))
        self.x = torch.cat([x_num, x_cat], 1)
        self.y = torch.unsqueeze(y, 1)

    def download(self, data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            for url in self.resources:
                data = requests.get(url).content
                filename = os.path.join(data_dir, os.path.basename(url))
                with open(filename, 'wb') as file:
                    file.write(data)


class MyDataLoader(BaseDataLoader):
    pass
