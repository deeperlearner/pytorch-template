import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class CNN_Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class VGG16(nn.Module):
    def __init__(self, feature_extracting, use_pretrained=True, num_classes=50):
        super().__init__()
        # VGG16 as feature extractor
        self.feature_extractor = vgg16(pretrained=use_pretrained)
        if feature_extracting:
            for param in self.feature_extractor.features.parameters():
                param.requires_grad = False
        self.feature_extractor.classifier[3] = nn.Linear(4096, 4096)
        self.feature_extractor.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

    def forward_nolast(self, x):
        x = self.feature_extractor.features(x)
        x = self.feature_extractor.avgpool(x)
        x = x.view(-1, 512*7*7)
        new_classifier = self.feature_extractor.classifier[:-3]
        x = new_classifier(x)
        return x
