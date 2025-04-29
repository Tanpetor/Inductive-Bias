import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        
        return x * attention

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
        
        self.attention = SpatialAttention()
        
        self.block3 = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8)
        )
        
        self.block4 = nn.Sequential(
            nn.Linear(in_features=8192, out_features=1024),
            nn.ReLU(),
        )
        
        self.attention_features = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SpatialAttention(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
        )
        
        self.fc = nn.Linear(1024 + 512, n_classes)
        
    def forward(self, x):
        path1 = self.block3(self.block1(x) + self.block2(x))
        path1 = torch.flatten(path1, start_dim=1)
        path1 = self.block4(path1)
        
        x_att = self.attention(self.block1(x) + self.block2(x))
        path2 = self.attention_features(x_att)
        
        combined = torch.cat([path1, path2], dim=1)
        
        out = self.fc(combined)
        return out