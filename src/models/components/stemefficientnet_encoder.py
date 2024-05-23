import torch
import torchvision.models as models
from torch import nn

from src.models.components.convbnrelu import ConvBNReLU


class StemEfficientNetEncoder(nn.Module):
    """EfficientNet (B0): https://arxiv.org/pdf/1905.11946.pdf suited to be UNet network encoder"""

    def __init__(self):
        """Constructs encoder from EfficientNetB0."""
        super().__init__()
        encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        features = encoder.features
        num_of_sequences = len(features)
        self.layer0 = features[: num_of_sequences - 6]
        self.layer1 = features[num_of_sequences - 6]
        self.layer2 = features[num_of_sequences - 5]
        self.layer3 = features[num_of_sequences - 4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns skip connections features from Efficient Net."""
        x = self.layer0(x)  # 3x224x224 -> 24x56x56
        x1 = self.layer1(x)  # 24x56x56 -> 40x28x28
        x2 = self.layer2(x1)  # 40x28x28 -> 80x14x14
        x3 = self.layer3(x2)  # 80x14x14 -> 112x14x14
        return x, x1, x2, x3
