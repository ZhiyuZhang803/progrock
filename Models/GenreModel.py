import torch.nn as nn


class GenreModel(nn.Module):
    def __init__(self):
        super(GenreModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=43, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.layer6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.layer7 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.layer8 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=10, kernel_size=1, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(nn.Linear(in_features=40, out_features=2), nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out
