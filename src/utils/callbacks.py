import io

import lightning.pytorch as lp
import torch
from PIL import Image

from src.utils.torch_utils import get_plot_vals


def get_pil_image_from_tensors(inputs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor) -> Image:
    """Returns PIL Image representing input/target/prediction images."""
    plt_fig = get_plot_vals(images=inputs.cpu(), targets=targets.cpu(), predictions=preds.cpu())
    buf = io.BytesIO()
    plt_fig.savefig(buf)
    buf.seek(0)
    image = Image.open(buf)
    return image


class LogPredictionSamplesCallback(lp.Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            if batch[0].shape[0] < 6:
                return
            images, targets = batch
            image = get_pil_image_from_tensors(inputs=images, targets=targets, preds=outputs)
            trainer.logger.log_image(key="samples", images=[image])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            if batch[0].shape[0] < 6:
                return
            images, targets = batch
            image = get_pil_image_from_tensors(inputs=images, targets=targets, preds=outputs)
            trainer.logger.log_image(key="samples", images=[image])
