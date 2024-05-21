import torch
import torch.nn as nn
import torch.nn.functional as F


class Res1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(residual)
        out += residual
        out = self.leakyrelu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, input_channels=43, num_classes=2):
        super(ResNet1D, self).__init__()

        # Initial Convolution
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)

        # Res1D + MaxPool blocks
        self.layer1 = self._make_layer(64, 64, stride=1, num_blocks=1)
        self.layer2 = self._make_layer(64, 64, stride=1, num_blocks=1)
        self.layer3 = self._make_layer(64, 128, stride=1, num_blocks=1)

        # Final Conv1D before output
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.maxpool = nn.MaxPool1d(1)
        self.fc = nn.Linear(7040, num_classes)

    def _make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        layers.append(Res1DBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(Res1DBlock(out_channels, out_channels, stride=1))
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.leaky_relu(self.conv2(x))
        # x = self.maxpool(x)
        x = torch.flatten(x, 1)  # Flatten the features for the fully connected layer
        x = self.fc(x)
        return x
