import torch
import torch.nn as nn
import torch.nn.functional as F

class ADDAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 20, kernel_size=5, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(20, 50, kernel_size=5, stride=1),
                nn.Dropout2d(),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.ReLU(inplace=True)
                )
        self.fc = nn.Linear(800, 500)

    def forward(self, x):
        x = self.feature_extractor(x)
        x_flat = x.view(-1, 800)
        output = self.fc(x_flat)
        return output

class ADDAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.clf = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(500, 10)
                )
    def forward(self, x):
        prob = self.clf(x)
        return prob

class ADDADiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Sequential(
                nn.Linear(500, 500),
                nn.ReLU(inplace=True),
                nn.Linear(500, 2),
                )

    def forward(self, x):
        prob = self.dis(x)
        return prob

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
