import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.ReLU(inplace=True)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 50, kernel_size=5),
                nn.BatchNorm2d(50),
                nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(inplace=True)
                )
        self.class_clf = nn.Sequential(
                nn.Linear(50*4*4, 100),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(100, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 10),
                )
        self.domain_clf = nn.Sequential(
                nn.Linear(50*4*4, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 2)
                )

    def forward(self, x, constant):
        x = self.conv1(x)
        x = self.conv2(x)
        x_flat = x.view(-1, 50*4*4)
        cls_output = self.class_clf(x_flat)
        x_input = GradReverse.apply(x_flat, constant)
        dom_output = self.domain_clf(x_input)
        return cls_output, dom_output
