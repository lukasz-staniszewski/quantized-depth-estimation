from torch import nn

from src.models.components.ffnet_decoder import DepthHeadC, UpC
from src.models.components.stemefficientnet_encoder import StemEfficientNetEncoder


class EfficientFFNetC(nn.Module):
    def __init__(self):
        """Encoder-Decoder U-Net type architecture containing EfficientNetB0 encoder and FFNet
        decoder."""
        super(EfficientFFNetC, self).__init__()
        self.encoder = StemEfficientNetEncoder()
        self.decoder = UpC([24, 40, 80, 112], [128, 16, 16, 16])
        self.head = DepthHeadC(176)

    def forward(self, x):
        """Forward method for EfficientFFNetC."""
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x
