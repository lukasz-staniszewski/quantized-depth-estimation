import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import Normalize, Resize


class UnNormalize(Normalize):
    def __init__(self, *args, **kwargs):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


@torch.no_grad()
def get_plot_vals(images: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor):
    def get_colored_depth_mask(depth_mask: np.ndarray) -> np.ndarray:
        mask = np.squeeze(depth_mask)
        d_min, d_max = np.min(mask), np.max(mask)
        return 255 * plt.cm.inferno(1 - ((mask - d_min) / (d_max - d_min)))[:, :, :3]

    plt.ioff()
    fig = plt.figure(figsize=(8, 2), dpi=200)
    for idx in range(6):
        ax = plt.subplot(2, 3, idx + 1)
        image = images[idx]
        image_size = (image.shape[1], image.shape[2])
        image, mask_target, mask_pred = (
            UnNormalize()(image),
            Resize(image_size)(targets[idx]),
            Resize(image_size)(predictions[idx]),
        )
        image, mask_target, mask_pred = (
            image.permute(1, 2, 0).numpy(),
            mask_target.permute(1, 2, 0).numpy(),
            mask_pred.permute(1, 2, 0).numpy(),
        )
        mask_colored_target = get_colored_depth_mask(depth_mask=mask_target)
        mask_colored_pred = get_colored_depth_mask(depth_mask=mask_pred)
        pixel_break = np.ones(shape=(mask_colored_pred.shape[0], 5, mask_colored_pred.shape[2])) * 255
        image = 255 * image
        plt_image = np.hstack([image, pixel_break, mask_colored_target, pixel_break, mask_colored_pred])
        plt.axis("off")
        plt.title("Image / target / prediction")
        ax.title.set_size(5)
        plt.imshow(plt_image.astype("uint8"))
    plt.ion()
    return fig
