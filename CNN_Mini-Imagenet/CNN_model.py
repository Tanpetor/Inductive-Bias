import torch
import torch.nn as nn


class BasicBlockNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_0 = nn.Conv2d(3, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.conv2d_res = nn.Conv2d(3, 64, kernel_size=1)
        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(in_features=4096, out_features=100)

    def forward(self, x):
        main_x = self.bn2(self.conv_2(self.relu(self.bn1(self.conv_1(x)))))
        add_x = self.conv_0(x)
        out_ = self.pool(self.relu(main_x + add_x))
        out = self.linear(torch.flatten(out_, start_dim=1))

        return out
