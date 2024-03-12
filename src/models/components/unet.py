import segmentation_models_pytorch as smp
import torch.nn as nn


class UNet(nn.Module):
    """Simple UNet model using segmentation_models_pytorch."""

    def __init__(self, trainable_encoder: bool = False):
        """UNet constructor.

        Args:
            trainable_encoder (bool, optional): if True, encoder of UNet is also trained. Defaults to False.
        """
        super().__init__()
        self.model = smp.UnetPlusPlus(encoder_name="resnext50_32x4d", in_channels=3, classes=1)
        self.trainable_encoder(trainable=trainable_encoder)

    def trainable_encoder(self, trainable: bool):
        """Decides if encoder is trainable."""
        for p in self.model.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        "Forward method for UNet"
        return self.model(x)

    def _num_params(
        self,
    ):
        """Returns number of params."""
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])
