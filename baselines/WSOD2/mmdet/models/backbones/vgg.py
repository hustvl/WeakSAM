import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ..builder import BACKBONES

@BACKBONES.register_module()
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)

        self.conv1_1.requires_grad = False
        self.conv1_2.requires_grad = False
        self.conv2_1.requires_grad = False
        self.conv2_2.requires_grad = False

    def forward(self, x):
        x = F.relu(self.conv1_1(x), inplace=True)
        x = F.relu(self.conv1_2(x), inplace=True)
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x), inplace=True)
        x = F.relu(self.conv2_2(x), inplace=True)
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x), inplace=True)
        x = F.relu(self.conv3_2(x), inplace=True)
        x = F.relu(self.conv3_3(x), inplace=True)
        x = self.pool3(x)
        x = F.relu(self.conv4_1(x), inplace=True)
        x = F.relu(self.conv4_2(x), inplace=True)
        x = F.relu(self.conv4_3(x), inplace=True)
        x = F.relu(self.conv5_1(x), inplace=True)
        x = F.relu(self.conv5_2(x), inplace=True)
        x = F.relu(self.conv5_3(x), inplace=True)
        return [x]

    def init_weights(self, pretrained=None):
        pass
