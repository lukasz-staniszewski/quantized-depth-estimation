import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.convbnrelu import ConvBNReLU


class AdapterConv(nn.Module):
    """AdapterConv for FFNet."""

    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=[64, 128, 256, 512]):
        super(AdapterConv, self).__init__()
        assert len(in_channels) == len(out_channels), "Number of input and output branches should match"
        self.adapter_conv = nn.ModuleList()

        for k in range(len(in_channels)):
            self.adapter_conv.append(
                ConvBNReLU(in_channels[k], out_channels[k], ks=1, stride=1, padding=0),
            )

    def forward(self, x):
        """Forward method for AdapterConv."""
        out = []
        for k in range(len(self.adapter_conv)):
            out.append(self.adapter_conv[k](x[k]))
        return out


class UpsampleCat(nn.Module):
    """UpsampleCat for FFNet."""

    def __init__(self, upsample_kwargs={"mode": "bilinear", "align_corners": True}):
        super(UpsampleCat, self).__init__()
        self._up_kwargs = upsample_kwargs

    def forward(self, x):
        """Upsample and concatenate feature maps."""
        assert isinstance(x, list) or isinstance(x, tuple)
        x0 = x[0]
        _, _, H, W = x0.size()
        for i in range(1, len(x)):
            x0 = torch.cat([x0, F.interpolate(x[i], (H, W), **self._up_kwargs)], dim=1)
        return x0


class UpBranch(nn.Module):
    """UpBranch in FFNet."""

    def __init__(
        self,
        in_channels=[64, 128, 256, 512],
        out_channels=[128, 128, 128, 128],
        upsample_kwargs={"mode": "bilinear", "align_corners": True},
    ):
        super(UpBranch, self).__init__()

        self._up_kwargs = upsample_kwargs

        self.fam_32_sm = ConvBNReLU(in_channels[3], out_channels[3], ks=3, stride=1, padding=1)
        self.fam_32_up = ConvBNReLU(in_channels[3], in_channels[2], ks=1, stride=1, padding=0)
        self.fam_16_sm = ConvBNReLU(in_channels[2], out_channels[2], ks=3, stride=1, padding=1)
        self.fam_16_up = ConvBNReLU(in_channels[2], in_channels[1], ks=1, stride=1, padding=0)
        self.fam_8_sm = ConvBNReLU(in_channels[1], out_channels[1], ks=3, stride=1, padding=1)
        self.fam_8_up = ConvBNReLU(in_channels[1], in_channels[0], ks=1, stride=1, padding=0)
        self.fam_4 = ConvBNReLU(in_channels[0], out_channels[0], ks=3, stride=1, padding=1)

        self.high_level_ch = sum(out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        """Forward method for UpBranch."""
        feat4, feat8, feat16, feat32 = x

        smfeat_32 = self.fam_32_sm(feat32)
        upfeat_32 = self.fam_32_up(feat32)

        _, _, H, W = feat16.size()
        x = F.interpolate(upfeat_32, (H, W), **self._up_kwargs) + feat16
        smfeat_16 = self.fam_16_sm(x)
        upfeat_16 = self.fam_16_up(x)

        _, _, H, W = feat8.size()
        x = F.interpolate(upfeat_16, (H, W), **self._up_kwargs) + feat8
        smfeat_8 = self.fam_8_sm(x)
        upfeat_8 = self.fam_8_up(x)

        _, _, H, W = feat4.size()
        smfeat_4 = self.fam_4(F.interpolate(upfeat_8, (H, W), **self._up_kwargs) + feat4)

        return smfeat_4, smfeat_8, smfeat_16, smfeat_32


class UpC(nn.Module):
    """C-type head for FFNet."""

    def __init__(self, in_channels, base_channels):
        super(UpC, self).__init__()
        self.layers = nn.Sequential(
            AdapterConv(in_channels, base_channels),
            UpBranch(
                in_channels=base_channels,
                out_channels=[128, 16, 16, 16],
                upsample_kwargs={"mode": "bilinear", "align_corners": True},
            ),
            UpsampleCat(upsample_kwargs={"mode": "bilinear", "align_corners": True}),
        )

    def forward(self, x):
        """Forward method for UpC."""
        return self.layers(x)


class DepthHeadC(nn.Module):
    """Depth task head for FFNet."""

    def __init__(self, in_channels):
        super(DepthHeadC, self).__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(in_channels, 128, ks=3, stride=1, padding=1),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        """Forward method for DepthHeadC."""
        return self.layers(x)
