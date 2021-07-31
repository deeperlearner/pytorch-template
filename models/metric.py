from math import sqrt

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

smooth = 1e-6


class MetricTracker:
    def __init__(self, keys_iter: list, keys_epoch: list, writer=None):
        self.writer = writer
        self.iter_record = pd.DataFrame(
            index=keys_iter,
            columns=[
                "current",
                "sum",
                "square_sum",
                "counts",
                "mean",
                "square_avg",
                "std",
            ],
            dtype=np.float64,
        )
        self.epoch_record = pd.DataFrame(
            index=keys_epoch, columns=["mean"], dtype=np.float64
        )
        self.reset()

    def reset(self):
        for col in self.iter_record.columns:
            self.iter_record[col].values[:] = 0

    def iter_update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.iter_record.at[key, "current"] = value
        self.iter_record.at[key, "sum"] += value * n
        self.iter_record.at[key, "square_sum"] += value * value * n
        self.iter_record.at[key, "counts"] += n

    def epoch_update(self, key, value):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.epoch_record.at[key, "mean"] = value

    def current(self):
        return dict(self.iter_record["current"])

    def avg(self):
        for key, row in self.iter_record.iterrows():
            self.iter_record.at[key, "mean"] = row["sum"] / row["counts"]
            self.iter_record.at[key, "square_avg"] = row["square_sum"] / row["counts"]

    def std(self):
        for key, row in self.iter_record.iterrows():
            self.iter_record.at[key, "std"] = sqrt(
                row["square_avg"] - row["mean"] ** 2 + smooth
            )

    def result(self):
        self.avg()
        self.std()
        iter_result = self.iter_record[["mean", "std"]]
        epoch_result = self.epoch_record
        return pd.concat([iter_result, epoch_result])


###################
# pick thresholds #
###################
THRESHOLD = 0.5


def Youden_J(target, output, beta=1.0):
    global THRESHOLD
    with torch.no_grad():
        y_true = target.cpu().numpy()
        y_score = output.cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        THRESHOLD = thresholds[np.argmax(beta * tpr - fpr)]
    return THRESHOLD


def F_beta(target, output, beta=1.0):
    global THRESHOLD
    with torch.no_grad():
        y_true = target.cpu().numpy()
        probas_pred = output.cpu().numpy()
        ppv, tpr, thresholds = precision_recall_curve(y_true, probas_pred)
        THRESHOLD = thresholds[np.argmax(beta * tpr - ppv)]
    return THRESHOLD


#############################
# for binary classification #
#############################
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
        y_true = target.cpu().numpy()
        y_pred = predict.cpu().numpy()
        value = recall_score(y_true, y_pred)
    return value


# precision
def PPV(target, output):
    with torch.no_grad():
        predict = (output > THRESHOLD).type(torch.uint8)
        y_true = target.cpu().numpy()
        y_pred = predict.cpu().numpy()
        value = precision_score(y_true, y_pred)
    return value


def F_beta_score(target, output, beta=1.0):
    recall = TPR(target, output)
    precision = PPV(target, output)
    score = (precision * recall) / (beta ** 2 * precision + recall)
    return score


# AUC
def AUROC(target, output):
    with torch.no_grad():
        y_true = target.cpu().numpy()
        y_score = output.cpu().numpy()
        value = roc_auc_score(y_true, y_score)
    return value


def AUPRC(target, output):
    with torch.no_grad():
        y_true = target.cpu().numpy()
        y_score = output.cpu().numpy()
        value = average_precision_score(y_true, y_score)
    return value


#################################
# for multiclass classification #
#################################
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
    """
    Compute mean IoU score over 6 classes
    """
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
