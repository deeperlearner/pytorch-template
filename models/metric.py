from math import sqrt

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve

smooth = 1e-6


class MetricTracker:
    def __init__(self, keys_iter: list, keys_epoch: list, writer=None):
        self.writer = writer
        self.metrics_iter = pd.DataFrame(index=keys_iter, columns=['current', 'sum', 'square_sum', 'counts',
                                                                   'mean', 'square_avg', 'std'])
        self.metrics_epoch = pd.DataFrame(index=keys_epoch, columns=['mean'])
        self.reset()

    def reset(self):
        for col in self.metrics_iter.columns:
            self.metrics_iter[col].values[:] = 0

    def iter_update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.metrics_iter.at[key, 'current'] = value
        self.metrics_iter.at[key, 'sum'] += value * n
        self.metrics_iter.at[key, 'square_sum'] += value * value * n
        self.metrics_iter.at[key, 'counts'] += n

    def epoch_update(self, key, value):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.metrics_epoch.at[key, 'mean'] = value

    def current(self):
        return dict(self.metrics_iter['current'])

    def avg(self):
        for key, row in self.metrics_iter.iterrows():
            self.metrics_iter.at[key, 'mean'] = row['sum'] / row['counts']
            self.metrics_iter.at[key, 'square_avg'] = row['square_sum'] / row['counts']

    def std(self):
        for key, row in self.metrics_iter.iterrows():
            self.metrics_iter.at[key, 'std'] = sqrt(row['square_avg'] - row['mean']**2 + smooth)

    def result(self):
        self.avg()
        self.std()
        iter_result = self.metrics_iter[['mean', 'std']]
        epoch_result = self.metrics_epoch
        return pd.concat([iter_result, epoch_result])


# for binary classification
THRESHOLD = 0.5
def Youden_J(target, output):
    global THRESHOLD
    with torch.no_grad():
        fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output.cpu().numpy())
        THRESHOLD = thresholds[np.argmax(tpr - fpr)]
    return THRESHOLD


def binary_accuracy(target, output):
    with torch.no_grad():
        predict = (output > THRESHOLD).type(torch.uint8)
        correct = 0
        correct += torch.sum(predict == target).item()
    return correct / len(target)


# recall
def TPR(target, output):
    with torch.no_grad():
        predict = (output > THRESHOLD).type(torch.uint8)
        value = recall_score(target.cpu().numpy(), predict.cpu().numpy())
    return value


# precision
def PPV(target, output):
    with torch.no_grad():
        predict = (output > THRESHOLD).type(torch.uint8)
        value = precision_score(target.cpu().numpy(), predict.cpu().numpy())
    return value


def AUROC(target, output):
    with torch.no_grad():
        predict = (output > THRESHOLD).type(torch.uint8)
        value = roc_auc_score(target.cpu().numpy(), predict.cpu().numpy())
    return value


def AUPRC(target, output):
    with torch.no_grad():
        predict = (output > THRESHOLD).type(torch.uint8)
        value = average_precision_score(target.cpu().numpy(), predict.cpu().numpy())
    return value


# for multiclass classification
def accuracy(target, output):
    with torch.no_grad():
        predict = torch.argmax(output, dim=1)
        assert predict.shape[0] == len(target)
        correct = 0
        correct += torch.sum(predict == target).item()
    return correct / len(target)


def top_k_acc(target, output, k=3):
    with torch.no_grad():
        predict = torch.topk(output, k, dim=1)[1]
        assert predict.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(predict[:, i] == target).item()
    return correct / len(target)


def mean_iou_score(target, output):
    '''
    Compute mean IoU score over 6 classes
    '''
    with torch.no_grad():
        predict = torch.argmax(output, dim=1)
        predict = predict.cpu().numpy()
        target = target.cpu().numpy()
        mean_iou = 0
        for i in range(6):
            tp_fp = np.sum(predict == i)
            tp_fn = np.sum(target == i)
            tp = np.sum((predict == i) * (target == i))
            iou = (tp + smooth) / (tp_fp + tp_fn - tp + smooth)
            mean_iou += iou / 6
    return mean_iou
