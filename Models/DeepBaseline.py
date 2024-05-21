"""
    Baseline model from example project (Fall 2022)

"""

import torch.nn as nn


class DeepBaseline(nn.Module):
    def __init__(self):
        super(DeepBaseline, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=43, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=13312, out_features=100), nn.ReLU()
        )

        self.fc2 = nn.Sequential(nn.Linear(in_features=100, out_features=10), nn.ReLU())

        self.fc3 = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
