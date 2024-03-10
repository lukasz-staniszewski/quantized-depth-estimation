from __future__ import annotations

import torch
import torch.nn as nn
from typing import List



class ConvNet(nn.Module):
    def __init__(self, input_size: List[int] = [1, 28, 28], n_classes: int = 10):
        super().__init__()


        C, H, W = input_size[0], input_size[1], input_size[2]

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.dense = nn.Sequential(
            nn.Linear(in_features=256 * (H // 8) * (W // 8), out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    _ = ConvNet()
