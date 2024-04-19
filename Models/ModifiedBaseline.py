"""
    Baseline model from example project (Fall 2022)

"""

import torch.nn as nn


class ModifiedBaseline(nn.Module):
    def __init__(self):
        super(ModifiedBaseline, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=43, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=6848, out_features=600), nn.ReLU()
        )

        self.fc2 = nn.Sequential(nn.Linear(in_features=600, out_features=10), nn.ReLU())

        self.fc3 = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
