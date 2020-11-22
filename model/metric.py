import numpy as np
import torch


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

smooth = 1e-6
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

