from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .components.chipmultilabel import ChipMultilabelDataset


class ChipMultilabelModule(LightningDataModule):
    def __init__(self,
        metadata_file: str,
        chips_folder: str,
        min_ohe_count = 1,
        get_strlabels = False,
        chip_transforms = None,
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
        ):
   
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset = ChipMultilabelDataset(
            metadata_file = metadata_file,
            chips_folder = chips_folder,
            chip_transforms = chip_transforms,
            get_strlabels = get_strlabels,
            min_ohe_count = min_ohe_count,
            split="train",
        )

        self.val_dataset = ChipMultilabelDataset(
            metadata_file = metadata_file,
            chips_folder = chips_folder,
            chip_transforms=chip_transforms,
            get_strlabels = get_strlabels,
            min_ohe_count = min_ohe_count,
            split="val",
        )

        self.test_dataset = ChipMultilabelDataset(
            metadata_file = metadata_file,
            chips_folder = chips_folder,
            chip_transforms = chip_transforms,
            get_strlabels = get_strlabels,
            min_ohe_count = min_ohe_count,
            split="test",
        )


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=8,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=8,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=8,
        )