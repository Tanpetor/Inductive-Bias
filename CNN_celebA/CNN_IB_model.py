import torch
import torch.nn as nn

class InductiveBiasCelebA(nn.Module):
    def __init__(self, n_classes=15):
        super().__init__()

        self.base_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.upper_face = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lower_face = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.global_features = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.local_features = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.gender_attention = self._make_attention_module(128)
        self.age_attention = self._make_attention_module(128)
        self.accessory_attention = self._make_attention_module(128)
        self.expression_attention = self._make_attention_module(128)

        self.upper_pool = nn.AdaptiveAvgPool2d(1)
        self.lower_pool = nn.AdaptiveAvgPool2d(1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d(1)

        upper_dim = 128
        lower_dim = 128
        global_dim = 128
        local_dim = 128
        total_dim = upper_dim + lower_dim + global_dim + local_dim

        self.gender_classifier = nn.Linear(total_dim, 2)
        self.age_classifier = nn.Linear(total_dim, 3)
        self.accessory_classifier = nn.Linear(total_dim, 3)

        self.classifier = nn.Sequential(
            nn.Linear(total_dim + 2 + 3 + 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def _make_attention_module(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        base = self.base_features(x)

        upper = self.upper_face(base)
        lower = self.lower_face(base)

        global_feat = self.global_features(base)
        local_feat = self.local_features(base)

        upper_gender = upper * self.gender_attention(upper)
        lower_gender = lower * self.gender_attention(lower)

        upper_age = upper * self.age_attention(upper)
        lower_age = lower * self.age_attention(lower)

        global_accessory = global_feat * self.accessory_attention(global_feat)

        lower_expression = lower * self.expression_attention(lower)

        upper_feat = self.upper_pool(upper_gender + upper_age).flatten(1)
        lower_feat = self.lower_pool(lower_gender + lower_expression).flatten(1)
        global_feat = self.global_pool(global_accessory).flatten(1)
        local_feat = self.local_pool(local_feat).flatten(1)

        combined_features = torch.cat([upper_feat, lower_feat, global_feat, local_feat], dim=1)

        gender_pred = self.gender_classifier(combined_features)
        age_pred = self.age_classifier(combined_features)
        accessory_pred = self.accessory_classifier(combined_features)

        final_features = torch.cat([
            combined_features,
            torch.softmax(gender_pred, dim=1),
            torch.softmax(age_pred, dim=1),
            torch.softmax(accessory_pred, dim=1)
        ], dim=1)

        output = self.classifier(final_features)

        return output
