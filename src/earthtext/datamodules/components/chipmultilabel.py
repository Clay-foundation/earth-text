import numpy as np
import pandas as pd
from loguru import logger
import torch
from torch.utils.data import Dataset
from progressbar import progressbar as pbar
from earthtext.io import io

class ChipMultilabelDataset(Dataset):

    """
    dataset to load chips with multilabel vector
    """

    def __init__(
        self,
        metadata_file: str,
        chips_folder: str,
        split: str,
        chip_transforms = None,
        get_strlabels = False
    ):

        """
        get_strlabels: return also the string description of the multilabels present
        """

        self.split = split
        self.chips_folder = chips_folder
        self.chip_transforms = chip_transforms
        self.metadata_file = metadata_file
        self.get_strlabels = get_strlabels
        self.metadata = io.read_multilabel_metadata(metadata_file)
        self.metadata = self.metadata[self.metadata['split']==split]
        nitems = len(self.metadata)
        # keep only the items for which there is actually a chip image file
        logger.info(f"checking chip files for {split} split")
        chips_exists = [io.check_chip_exists(chips_folder, i['col'], i['row']) for _, i in pbar(self.metadata.iterrows(), max_value=len(self.metadata))]
        self.metadata = self.metadata[chips_exists]
        logger.info(f"read {split} split with {len(self.metadata)} chip files (out of {nitems})")
        

    def prepare_data(self):
        """This is an optional preprocessing step to be defined in each dataloader.
        It will be called by the Pytorch Lighting Data Module.
        All the preprocessing steps such as normalization or transformations
        might be write in this method when you override it.
        """

    def __len__(self):
        return len(self.metadata)


    def __repr__(self):
        return f"{self.__class__.__name__} {self.split} split with {len(self)} items"


    def __getitem__(self, idx):

        item = self.metadata.iloc[idx]
        chip = io.read_chip(self.chips_folder, item['col'], item['row'])
        multilabel = item.onehot_label

        if self.chip_transforms is not None:
            chip = self.chip_transforms(chip)

        r = {
             'chip': chip, 
             'multilabel': torch.tensor(multilabel).type(torch.int8)
            }

        if self.get_strlabels:
            r['str_multilabel'] = " ".join( item.string_labels)

        return r

