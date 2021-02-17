import os

import requests
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

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
        self.len_data = len(train_data.index)

        test_path = os.path.join(data_dir, 'adult.test')
        test_data = pd.read_csv(test_path, sep=', ', names=columns, engine='python', skiprows=1, na_values='?')
        self.len_test = len(test_data.index)

        merged_data = pd.concat([train_data, test_data])
        self.data = merged_data.drop(["income"], axis=1)
        self.label = merged_data.loc[:, "income"]

        self.num_data = self.data.select_dtypes(include=['int'])
        self.cat_data = self.data.select_dtypes(include=['object'])
        self._preprocess()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        if self.mode == 'train':
            return self.len_data
        return self.len_test

    def _preprocess(self):
        self._simple_impute()
        self.label.replace(to_replace=r'<=50K.?', value=0, regex=True, inplace=True)
        self.label.replace(to_replace=r'>50K.?', value=1, regex=True, inplace=True)

        self.cat_data = pd.get_dummies(self.cat_data)

    def _simple_impute(self):
        # impute numerical data with mean
        self.num_data = self.num_data.fillna(self.num_data.mean())
        # impute categorical data with mode
        self.cat_data = self.cat_data.fillna(self.cat_data.mode().loc[0])

    def normalize(self, split_idx=None):
        # adult.data
        x_num_data = self.num_data.iloc[:self.len_data]
        x_cat_data = self.cat_data.iloc[:self.len_data]
        y_data = self.label.iloc[:self.len_data]
        # adult.test
        if self.mode == 'test':
            x_num_test = self.num_data.iloc[self.len_data:]
            x_cat_test = self.cat_data.iloc[self.len_data:]
            y_test = self.label.iloc[self.len_data:]

        if self.mode == 'train':
            train_idx, valid_idx = split_idx
            # compute mean, std on training data only!
            mean = x_num_data.iloc[train_idx].mean()
            std = x_num_data.iloc[train_idx].mean()

            x_num, x_cat = x_num_data, x_cat_data
            y = y_data
        elif self.mode == 'test':
            # compute mean, std on adult.data
            mean = x_num_data.mean()
            std = x_num_data.mean()

            x_num, x_cat = x_num_test, x_cat_test
            y = y_test

        # normalize on numerical data only!
        x_num = (x_num - mean) / std
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
