from typing import List, Tuple

import numpy as np
from pandas import DataFrame
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class NYUDataset(Dataset):
    """NYU Depth V2 Dataset from https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2"""

    def __init__(self, df: DataFrame, tfms: List, mask_final_size: Tuple[int, int]):
        super().__init__()
        """Constructor of the NYUDataset class."""
        self.df = df
        self.tfms = tfms
        self.mask_final_size = mask_final_size

    def open_im(self, path: str):
        """Opens image from specific path.

        If gray flag is set to True, the image is converted to grayscale.
        """
        im = Image.open(str(path))
        im = np.array(im)
        return im

    def open_mask(self, path: str):
        """Opens mask from specific path."""
        mask = Image.open(str(path)).convert("L")
        mask = np.array(mask)
        return mask

    def __len__(
        self,
    ):
        """Returns size of the datset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Standard __getitem__ method for torch Dataset."""
        sample = self.df.iloc[idx, :]
        image, depth_image = self.open_im(path=sample[0]), self.open_mask(path=sample[1])
        augs = self.tfms(image=image, mask=depth_image)
        image, depth_image = augs["image"], augs["mask"] / 255
        depth_image = T.Resize(size=self.mask_final_size)(depth_image.unsqueeze(0))
        return image, depth_image


class NYUDatasetTest(NYUDataset):
    """NYU Depth V2 Test Dataset from https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2"""

    def open_mask(self, path: str) -> np.ndarray:
        """Opens mask from specific path.

        Test set masks contain centimeters, hence we convert it to numbers from 0 to 1.
        """
        mask = Image.open(str(path))
        mask = np.array(mask, np.int32)
        mask = ((mask / 10_000) * 255).astype(np.uint8)  # convert to [0,255]
        return mask
