import torch
from torch import nn

from src.models.components.bifpn_decoder import BiFPNDecoder
from src.models.components.effnet_decoder import EfficientNetEncoder


class EffNetBiFPN(nn.Module):
    """Connects EfficientNet type encoder with BiFPN type decoder and appends Conv head for depth
    estimation."""

    def __init__(self):
        super().__init__()
        self.encoder = EfficientNetEncoder()
        self.decoder_1 = BiFPNDecoder([24, 40, 80])
        self.decoder_2 = BiFPNDecoder([64, 64, 64])
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="nearest")
        self.upsample_2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=64 * 3, out_channels=1, kernel_size=3, padding="same"),
            nn.Upsample(scale_factor=4, mode="nearest"),
        )

    def forward(self, x):
        """Forwards EffNetBiFPN."""
        p4, p5, p6 = self.encoder.get_features(x)
        out_dec_1 = self.decoder_1([p4, p5, p6])
        out_dec_2 = self.decoder_2(out_dec_1)
        cat_out = self.concatenate_bifpn_features(out_dec_2)
        conv_out = self.final_convolution(cat_out)
        return self.final_upsample(conv_out)

    def concatenate_bifpn_features(self, out_bfpn3):
        """Concatenates features coming from BiFPN."""
        p4 = out_bfpn3[0]
        p5 = out_bfpn3[1]
        p6 = out_bfpn3[2]
        p6_up = self.upsample_4(p6)
        p5_up = self.upsample_2(p5)
        return torch.cat([p4, p5_up, p6_up], dim=1)
