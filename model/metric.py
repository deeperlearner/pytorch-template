from math import sqrt

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


smooth = 1e-6
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys,
            columns=['current', 'sum', 'square_sum', 'counts',
                'mean', 'square_avg', 'std'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.at[key, 'current'] = value
        self._data.at[key, 'sum'] += value * n
        self._data.at[key, 'square_sum'] += value * value * n
        self._data.at[key, 'counts'] += n

    def current(self):
        return dict(self._data['current'])

    def avg(self):
        for key, row in self._data.iterrows():
            self._data.at[key, 'mean'] = row['sum'] / row['counts']
            self._data.at[key, 'square_avg'] = row['square_sum'] / row['counts']

    def std(self):
        for key, row in self._data.iterrows():
            self._data.at[key, 'std'] = sqrt(row['square_avg']-row['mean']**2 + smooth)

    def result(self):
        self.avg()
        self.std()
        return self._data[['mean', 'std']]

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def auc(output, target):
    with torch.no_grad():
        value = roc_auc_score(target.cpu().numpy(), output.cpu().numpy())
    return value

def mean_iou_score(output, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        pred = pred.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        mean_iou = 0
        for i in range(6):
            tp_fp = np.sum(pred == i)
            tp_fn = np.sum(labels == i)
            tp = np.sum((pred == i) * (labels == i))
            iou = (tp + smooth) / (tp_fp + tp_fn - tp + smooth)
            mean_iou += iou / 6

    return mean_iou
