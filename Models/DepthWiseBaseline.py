"""
    Baseline model from example project (Fall 2022)

"""

import torch.nn as nn


class DepthWiseBaseline(nn.Module):
    def __init__(self):
        super(DepthWiseBaseline, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=43, out_channels=86, kernel_size=3, padding=1),
            nn.BatchNorm1d(86),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=86, out_channels=172, kernel_size=3, padding=1),
            nn.BatchNorm1d(172),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=172, out_channels=344, kernel_size=3, padding=1),
            nn.BatchNorm1d(344),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=18232, out_features=2), nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out
