# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/21 20:10 
# @Author : wzy 
# @File : model.py
# ---------------------------------------
import torch
import torch.nn as nn
from torchinfo import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4 * 4 * 64, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # 很关键，之前拉开用的view，在剪枝时不会被压缩，导致矩阵对不上，耗费大量时间！！！
        output = self.fc(x)
        return output


if __name__ == '__main__':
    model = Net()
    print(model)
    summary(model, input_size=(1, 3, 32, 32))
