import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


def nll_loss(output, target):
    return F.nll_loss(output, target)

CELoss = CrossEntropyLoss()
