import torch.nn as nn
import torch

class FoodClassifier(nn.Module):
    def __init__(self, n_classes=101):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32)
        )
        self.block2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)
        self.block3 = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8)
        )
        self.block4 = nn.Linear(in_features=8192, out_features=128)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        out = self.block3(self.block1(x) + self.block2(x))
        out = torch.flatten(out, start_dim=1)
        out = self.block4(out)
        out = self.fc(out)
        return out