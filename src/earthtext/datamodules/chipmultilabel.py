from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .components.chipmultilabel import ChipMultilabelDataset


class ChipMultilabelModule(LightningDataModule):
    def __init__(self,
        metadata_file: str,
        chips_folder: str = None,
        embeddings_folder: str = None,
        patch_embeddings_folder: str = None,
        multilabel_threshold_osm_ohecount = None,
        multilabel_threshold_osm_ohearea = None,
        get_osm_strlabels = False,
        get_osm_ohecount = False,
        get_osm_ohearea = False,
        get_osm_ohelength = False,
        get_esawc_proportions = False,
        get_chip_id = False,
        max_items = None,
        embeddings_normalization = None,
        normalize_input = False,
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
            embeddings_folder = embeddings_folder,
            patch_embeddings_folder = patch_embeddings_folder,
            chip_transforms = chip_transforms,
            get_osm_strlabels = get_osm_strlabels,
            get_osm_ohecount = get_osm_ohecount,
            get_osm_ohearea = get_osm_ohearea,
            get_osm_ohelength = get_osm_ohelength,
            get_esawc_proportions = get_esawc_proportions,
            get_chip_id = get_chip_id,
            max_items = max_items,
            embeddings_normalization = embeddings_normalization,
            normalize_input = normalize_input,
            multilabel_threshold_osm_ohecount = multilabel_threshold_osm_ohecount,
            multilabel_threshold_osm_ohearea = multilabel_threshold_osm_ohearea,
            split="train",
        )

        self.val_dataset = ChipMultilabelDataset(
            metadata_file = metadata_file,
            chips_folder = chips_folder,
            embeddings_folder = embeddings_folder,
            patch_embeddings_folder = patch_embeddings_folder,
            chip_transforms=chip_transforms,
            get_osm_strlabels = get_osm_strlabels,
            get_osm_ohecount = get_osm_ohecount,
            get_osm_ohearea = get_osm_ohearea,
            get_osm_ohelength = get_osm_ohelength,
            get_esawc_proportions = get_esawc_proportions,
            get_chip_id = get_chip_id,
            max_items = max_items,
            embeddings_normalization = embeddings_normalization,
            normalize_input = normalize_input,
            multilabel_threshold_osm_ohecount = multilabel_threshold_osm_ohecount,
            multilabel_threshold_osm_ohearea = multilabel_threshold_osm_ohearea,
            split="val",
        )

        self.test_dataset = ChipMultilabelDataset(
            metadata_file = metadata_file,
            chips_folder = chips_folder,
            embeddings_folder = embeddings_folder,
            patch_embeddings_folder = patch_embeddings_folder,
            chip_transforms = chip_transforms,
            get_osm_strlabels = get_osm_strlabels,
            get_osm_ohecount = get_osm_ohecount,
            get_osm_ohearea = get_osm_ohearea,
            get_osm_ohelength = get_osm_ohelength,
            get_esawc_proportions = get_esawc_proportions,
            get_chip_id = get_chip_id,
            max_items = max_items,
            embeddings_normalization = embeddings_normalization,
            normalize_input = normalize_input,
            multilabel_threshold_osm_ohecount = multilabel_threshold_osm_ohecount,
            multilabel_threshold_osm_ohearea = multilabel_threshold_osm_ohearea,
            split="test",
        )

    def disable_chip_loading(self):
        self.train_dataset.disable_chip_loading = True
        self.test_dataset.disable_chip_loading = True
        self.val_dataset.disable_chip_loading = True

    def enable_chip_loading(self):
        self.train_dataset.disable_chip_loading = False
        self.test_dataset.disable_chip_loading = False
        self.val_dataset.disable_chip_loading = False


    def train_dataloader(self, shuffle=True):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            persistent_workers=True,
            prefetch_factor=8,
        )

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            persistent_workers=True,
            prefetch_factor=8,
        )

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            persistent_workers=True,
            prefetch_factor=8,
        )