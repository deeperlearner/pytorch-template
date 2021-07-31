import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self.apply(weights_init)

    def forward(self, img):
        img = F.relu(F.max_pool2d(self.conv1(img), 2))
        img = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(img)), 2))
        img = img.view(-1, 320)
        img = F.relu(self.fc1(img))
        img = F.dropout(img, training=self.training)
        img = self.fc2(img)
        return F.log_softmax(img, dim=1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)
