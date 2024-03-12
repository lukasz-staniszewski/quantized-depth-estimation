from typing import List

import cv2 as cv
from pandas import DataFrame
from torch.utils.data import Dataset


class NYUDataset(Dataset):
    """NYU Depth V2 Dataset from https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2"""

    def __init__(self, df: DataFrame, tfms: List):
        super().__init__()
        """Constructor of the NYUDataset class."""
        self.df = df
        self.tfms = tfms

    def open_im(self, path: str, gray: bool = False):
        """Opens image from specific path.

        If gray flag is set to True, the image is converted to grayscale.
        """
        im = cv.imread(str(path))
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY if gray else cv.COLOR_BGR2RGB)
        return im

    def __len__(
        self,
    ):
        """Returns size of the datset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Standard __getitem__ method for torch Dataset."""
        s = self.df.iloc[idx, :]
        image, depth_image = s[0], s[1]
        image, depth_image = self.open_im(path=image), self.open_im(path=depth_image, gray=True)
        augs = self.tfms(image=image, mask=depth_image)
        image, depth_image = augs["image"], augs["mask"] / 255.0
        return image, depth_image.unsqueeze(0)
