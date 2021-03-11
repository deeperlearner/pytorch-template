import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, in_features=1):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

        self.weights_reset()

    def weights_reset(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
