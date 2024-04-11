import numpy as np
import pandas as pd
from loguru import logger
import torch
from torch.utils.data import Dataset
from progressbar import progressbar as pbar
from earthtext.io import io
torch.multiprocessing.set_sharing_strategy('file_system')

esawc_map = {
'10':	'Tree cover',
'20':	'Shrubland',
'30':	'Grassland',
'40':	'Cropland',
'50':	'Built-up',
'60':	'Bare / sparse vegetation',
'70':	'Snow and ice',
'80':	'Permanent water bodies',
'90':	'Herbaceous wetland',
'95':	'Mangroves',
'100':'Moss and lichen'
}

class ChipMultilabelDataset(Dataset):

    """
    dataset to load chips with multilabel vector
    """

    def __init__(
        self,
        metadata_file: str,
        split: str,
        chips_folder: str = None,
        embeddings_folder: str = None,
        patch_embeddings_folder: str = None,
        chip_transforms = None,
        get_strlabels = False,
        get_esawc_proportions = False,
        get_chip_id = False,
        min_ohe_count = 1,
        cache_size = -1
    ):

        """
        get_strlabels: return also the string description of the multilabels present
        min_ohe_count: returns a 0/1 ohe vector with 1 if the ohe count is > min_ohecount
                       if None, it will return the ohe count
        """

        self.split = split
        self.chips_folder = chips_folder
        self.chip_transforms = chip_transforms
        self.embeddings_folder = embeddings_folder
        self.patch_embeddings_folder = patch_embeddings_folder
        self.get_esawc_proportions = get_esawc_proportions
        self.metadata_file = metadata_file
        self.get_chip_id = get_chip_id
        self.get_strlabels = get_strlabels
        self.metadata = io.read_multilabel_metadata(metadata_file)
        self.metadata = self.metadata[self.metadata['split']==split]
        self.min_ohe_count = min_ohe_count
        nitems = len(self.metadata)
        # keep only the items for which there is actually a chip image file
        logger.info(f"checking chip files for {split} split")
        chips_exists = [io.check_chip_exists(chips_folder, embeddings_folder, patch_embeddings_folder, i['col'], i['row']) \
                               for _, i in pbar(self.metadata.iterrows(), max_value=len(self.metadata))]
        self.metadata = self.metadata[chips_exists]
        logger.info(f"read {split} split with {len(self.metadata)} chip files (out of {nitems})")
        
        logger.info(f"max cache size is {cache_size}")
        self.cache_size = cache_size
        self.cache = {}

    def prepare_data(self):
        """This is an optional preprocessing step to be defined in each dataloader.
        It will be called by the Pytorch Lighting Data Module.
        All the preprocessing steps such as normalization or transformations
        might be write in this method when you override it.
        """

    def __len__(self):
        return len(self.metadata)


    def __repr__(self):
        return f"{self.__class__.__name__} {self.split} split with {len(self)} items, in cache {len(self.cache)} items"


    def __getitem__(self, idx):

        if idx in self.cache.keys():
            return self.cache[idx]

        r = {}        

        item = self.metadata.iloc[idx]
        multilabel = item.onehot_count.astype(int)
        if self.min_ohe_count is not None:
            multilabel = (multilabel> self.min_ohe_count).astype(int)

        r['multilabel'] = torch.tensor(multilabel).type(torch.int8)

        if self.get_chip_id:
            r['chip_id'] = item.name

        if self.chips_folder is not None:
            chip = io.read_chip(self.chips_folder, item['col'], item['row'])
            if self.chip_transforms is not None:
                chip = self.chip_transforms(chip)
            r['chip'] = chip

        if self.embeddings_folder is not None:
            r['embedding'] = io.read_embedding(self.embeddings_folder,  item['col'], item['row'])

        if self.patch_embeddings_folder is not None:
            r['patch_embedding'] = io.read_patch_embedding(self.patch_embeddings_folder,  item['col'], item['row'])

        if self.get_strlabels:
            r['str_multilabel'] = " ".join( item.string_labels)

        if self.get_esawc_proportions:
            r['esawc_proportions'] = str(item.esawc_proportions)

        # store in cache
        if self.cache_size == -1 or len(self.cache) < self.cache_size:
            self.cache[idx] = r

        return r

