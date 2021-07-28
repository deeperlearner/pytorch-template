import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, in_features=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        output = self.linear(x)
        output = torch.sigmoid(output)
        return output
