import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim


class FRNet(nn.Module):
    def __init__(self, input_size, n_classes, num_fout1, num_fout2, num_fout3):
        super(FRNet, self).__init__()
        self.input_size = input_size
        self.num_fout3 = num_fout3

        # 訓練済みVGGの15層目までの読み込み
        self.vgg16_features = nn.ModuleList(list(torchvision.models.vgg16(pretrained=True).features)[:16])
        for p in self.vgg16_features.parameters():
            p.requires_grad = False

        # 層の定義
        self.c1 = nn.Conv2d(256, num_fout1, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(num_fout1, num_fout2, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(num_fout2, num_fout3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_fout1)
        self.bn2 = nn.BatchNorm2d(num_fout2)
        self.fc = nn.Linear(int(input_size/4*input_size/4*num_fout3),n_classes)

    def forward(self, x):
        with torch.no_grad():
            self.vgg16_features.eval()
            for f in self.vgg16_features:
                x = f(x)

        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu((self.bn2(self.c2(x))))
        x = self.c3(x)
        x = x.view(-1,int(self.input_size/4*self.input_size/4*self.num_fout3))
        x = self.fc(x)

        return x
