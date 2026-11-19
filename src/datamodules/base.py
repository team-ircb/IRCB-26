"""Shared data-module pieces: the base class and a dataset of aligned arrays."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
class DGNDataModuleBase(pl.LightningDataModule):
    """
    Base class for all Data Generating Networks (DGN) Lightning DataModules.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def setup(self, stage=None):
        raise NotImplementedError("setup must be implemented in subclass!")

    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.get("num_workers", 16),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=len(self.val_ds),
            shuffle=False,
            num_workers=self.hparams.get("num_workers", 16),
        )


class BasicDataset(Dataset):
    """
    Basic dataset class for Data Generating Networks (DGN).
    """
    def __init__(self, *iter_data, **kwargs):
        self.iter_data = iter_data
        self.info = kwargs

    def __getitem__(self, index):
        return tuple([item[index] for item in self.iter_data]), self.info

    def __len__(self):
        return len(self.iter_data[0])
