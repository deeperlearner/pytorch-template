import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class VGG16(nn.Module):
    def __init__(self, feature_extracting, use_pretrained=True, num_classes=50):
        super(VGG16, self).__init__()
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
        x = x.view(-1, 512 * 7 * 7)
        new_classifier = self.feature_extractor.classifier[:-3]
        x = new_classifier(x)
        return x
