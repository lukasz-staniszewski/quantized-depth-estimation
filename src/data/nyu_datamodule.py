import os
from typing import Any, Dict, Optional, Tuple

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.data.components.nyu_dataset import NYUDataset


class NYUDataModule(LightningDataModule):
    """`NYUDataModule` for the NYUv2 dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/nyu_data/",
        train_val_test_split: Tuple[float, float, float] = (0.85, 0.15, 0.0),
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = False,
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """Initialize a `FashionMNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = A.Compose(
            [
                A.GaussNoise(p=0.2),
                A.RGBShift(),
                A.RandomBrightnessContrast(),
                A.RandomResizedCrop(150, 150),
                A.ColorJitter(),
                A.Resize(input_size[0], input_size[1]),
                A.Normalize(always_apply=True),
                ToTensorV2(),
            ]
        )
        self.valid_tfms = A.Compose(
            [A.Resize(input_size[0], input_size[1]), A.Normalize(always_apply=True), ToTensorV2()]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.input_size = input_size
        self.mask_final_size = (int(input_size[0] / 4), int(input_size[1] / 4))

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_csv_path = os.path.join(self.hparams.data_dir, "data/nyu2_train.csv")
            test_csv_path = os.path.join(self.hparams.data_dir, "data/nyu2_test.csv")
            df_train = pd.read_csv(train_csv_path, header=None)
            df_test = pd.read_csv(test_csv_path, header=None)
            tqdm.pandas()
            df_train[0] = df_train[0].progress_map(lambda x: os.path.join(self.hparams.data_dir, x))
            df_train[1] = df_train[1].progress_map(lambda x: os.path.join(self.hparams.data_dir, x))
            df_test[0] = df_test[0].progress_map(lambda x: os.path.join(self.hparams.data_dir, x))
            df_test[1] = df_test[1].progress_map(lambda x: os.path.join(self.hparams.data_dir, x))

            trainset = NYUDataset(df=df_train, tfms=self.train_transforms, mask_final_size=self.mask_final_size)
            testset = NYUDataset(df=df_test, tfms=self.valid_tfms, mask_final_size=self.mask_final_size)

            self.data_train, self.data_val, _ = random_split(
                dataset=trainset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            self.data_val.tfms = self.valid_tfms
            self.data_test = testset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = NYUDataModule()
