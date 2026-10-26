"""WM-811K wafer-map classification models."""

import torch
from torch import nn


class SmallWaferCNN(nn.Module):
    """Compact CNN for the defect/valid two-channel wafer-map representation."""

    def __init__(self, num_classes=9):
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer greater than one")
        self.features = nn.Sequential(
            self._block(2, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):
        return self.classifier(self.pool(self.features(images)))
