import torch
import torchvision.models as models
from torch import nn


class EfficientNetEncoder(nn.Module):
    """EfficientNet (B0) type encoder: https://arxiv.org/pdf/1905.11946.pdf"""

    def __init__(self):
        """Constructs encoder from EfficientNetB0."""
        super().__init__()
        encoder = models.efficientnet_b0()
        features = encoder.features
        num_of_sequences = len(features)
        self.layer1 = features[: num_of_sequences - 6]
        self.layer2 = features[num_of_sequences - 6]
        self.layer3 = features[num_of_sequences - 5]

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns skip connections features from Efficient Net."""
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x1, x2, x3
