from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import os
from loguru import logger
from ..io import io
from .components.chipmultilabel import ChipMultilabelDataset
import pickle
from progressbar import progressbar as pbar
import numpy as np

class OSMandEmbeddingsNormalizer:
    """
    loops over the train part of a dataloader computing the means and stdevs
    of osmvectors and embeddings, and saving them to a file
    """

    def __init__(self, dataloader, force_compute=False):
        self.dataloader = dataloader
        self.train_dataset = dataloader.train_dataset

        self.has_file_embeddings = self.train_dataset.embeddings_folder is not None
        self.has_metadata_embeddings = self.train_dataset.metadata_has_embeddings
        
        self.force_compute = force_compute

        if self.has_file_embeddings:
            embeddings_name = self.train_dataset.embeddings_folder.split("/")[-1]
        elif self.has_metadata_embeddings:
            embeddings_name = "metadata_embeddings"
        else:
            embeddings_name = "no_embeddings"

        # the name of the constants file contains the metadata file name and the embeddings folder name
        # so that different embeddings and metadatafiles get their own constants
        self.means_stdevs_file = ".".join(self.train_dataset.metadata_file.split(".")[:-1])+"_"+embeddings_name+"_meansstdevs.pkl"                

        self.compute()
    
    def compute(self):
        if not self.force_compute and os.path.isfile(self.means_stdevs_file):
            logger.info(f"reading means and stddevs from {self.means_stdevs_file}")
            with open(self.means_stdevs_file, "rb") as f:
                self.constants = pickle.load(f)
            return

        logger.info("computing mean and stddevs for embeddings and osm vectors")

        embeddings, osm_counts, osm_areas, osm_lengths = [], [], [], []
        for _,item in pbar(self.train_dataset.metadata.iterrows(), max_value=len(self.train_dataset)):
            osm_counts.append(item.onehot_count)
            osm_areas.append(item.onehot_area)
            osm_lengths.append(item.onehot_length)
        
            if self.has_file_embeddings:
                embedding = io.read_embedding(self.train_dataset.embeddings_folder,  item['col'], item['row']) 
                embeddings.append(embedding)
            elif self.has_metadata_embeddings:
                embeddings.append(item['embeddings'])                        

        self.constants = {
            'means':{
                    'osm_counts':  np.mean(np.r_[osm_counts], axis=0),
                    'osm_areas':   np.mean(np.r_[osm_areas], axis=0),
                    'osm_lengths': np.mean(np.r_[osm_lengths], axis=0)
            },
            'stdevs':{
                    'osm_counts':  np.std(np.r_[osm_counts], axis=0) + 1e-5,
                    'osm_areas':   np.std(np.r_[osm_areas], axis=0) + 1e-5,
                    'osm_lengths': np.std(np.r_[osm_lengths], axis=0) + 1e-5
            }
        }

        if self.has_file_embeddings or self.has_metadata_embeddings:
            self.constants['means']['embeddings'] = np.mean(np.r_[embeddings], axis=0)
            self.constants['stdevs']['embeddings'] = np.std(np.r_[embeddings], axis=0)

        with open(self.means_stdevs_file, "wb") as f:
            pickle.dump(self.constants, f)
        logger.info(f"means and stddevs saved to {self.means_stdevs_file}")


    def normalize_embeddings(self, x):
        if not self.has_file_embeddings and not self.has_metadata_embeddings:
            raise ValueError("this normalizer object has no embeddings")
        _x = (x - self.constants['means']['embeddings']) / self.constants['stdevs']['embeddings']

        if x.ndim == 3:  # x is a neighbors 3D array. Masked normalization of each neighbor embedding.
            zero_mask = x.sum(axis=-1) == 0
            _x[zero_mask] = 0

        return _x

    def unnormalize_embeddings(self, x):
        if not self.has_file_embeddings and not self.has_metadata_embeddings:
            raise ValueError("this normalizer object has no embeddings")

        return  x * self.constants['stdevs']['embeddings'] - self.constants['means']['embeddings']

    def normalize_osm_vector_area(self, x):
        return  (x - self.constants['means']['osm_areas']) / self.constants['stdevs']['osm_areas']

    def normalize_osm_vector_count(self, x):
        return  (x - self.constants['means']['osm_counts']) / self.constants['stdevs']['osm_counts']

    def normalize_osm_vector_length(self, x):
        return  (x - self.constants['means']['osm_lengths']) / self.constants['stdevs']['osm_lengths']

    def unnormalize_osm_vector_area(self, x, indexes=None):
        if indexes is None:
            r =  x * self.constants['stdevs']['osm_areas'] + self.constants['means']['osm_areas']
        else:
            r =  x * self.constants['stdevs']['osm_areas'][indexes] + self.constants['means']['osm_areas'][indexes]
        return r.astype(int)

    def unnormalize_osm_vector_count(self, x, indexes=None):
        if indexes is None:
            r =   x * self.constants['stdevs']['osm_counts'] + self.constants['means']['osm_counts']
        else:
            r =   x * self.constants['stdevs']['osm_counts'][indexes] + self.constants['means']['osm_counts'][indexes]
        return r.astype(int)

    def unnormalize_osm_vector_length(self, x, indexes=None):
        if indexes is None:
            r =  x * self.constants['stdevs']['osm_lengths'] + self.constants['means']['osm_lengths']
        else:
            r =  x * self.constants['stdevs']['osm_lengths'][indexes] + self.constants['means']['osm_lengths'][indexes]
        return r.astype(int)

    def unnormalize_osm_vector(self, x, indexes=None):
        """
        x is a dict with keys 'osm_ohecount', 'osm_ohearea', 'osm_ohelength'
        """
        return {'osm_ohecount':  self.unnormalize_osm_vector_count(x['osm_ohecount'], indexes),
                'osm_ohearea':   self.unnormalize_osm_vector_area(x['osm_ohearea'], indexes),
                'osm_ohelength': self.unnormalize_osm_vector_length(x['osm_ohelength'], indexes)
        }


class ChipMultilabelModule(LightningDataModule):
    def __init__(self,
        metadata_file: str,
        chips_folder: str = None,
        embeddings_folder: str = None,
        patch_embeddings_folder: str = None,
        neighbor_embeddings_folder: str = None,
        multilabel_threshold_osm_ohecount = None,
        multilabel_threshold_osm_ohearea = None,
        get_osm_strlabels = False,
        get_osm_ohecount = False,
        get_osm_ohearea = False,
        get_osm_ohelength = False,
        get_esawc_proportions = False,
        get_chip_id = False,
        max_items = None,
        embeddings_normalization = True,
        osmvector_normalization = False,
        osm_codeset = 'sentinel2',
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
            neighbor_embeddings_folder = neighbor_embeddings_folder,
            chip_transforms = chip_transforms,
            get_osm_strlabels = get_osm_strlabels,
            get_osm_ohecount = get_osm_ohecount,
            get_osm_ohearea = get_osm_ohearea,
            get_osm_ohelength = get_osm_ohelength,
            get_esawc_proportions = get_esawc_proportions,
            get_chip_id = get_chip_id,
            max_items = max_items,
            embeddings_normalization = embeddings_normalization,
            osmvector_normalization = osmvector_normalization,
            osm_codeset = osm_codeset,
            multilabel_threshold_osm_ohecount = multilabel_threshold_osm_ohecount,
            multilabel_threshold_osm_ohearea = multilabel_threshold_osm_ohearea,
            split="train",
        )

        self.val_dataset = ChipMultilabelDataset(
            metadata_file = metadata_file,
            chips_folder = chips_folder,
            embeddings_folder = embeddings_folder,
            patch_embeddings_folder = patch_embeddings_folder,
            neighbor_embeddings_folder = neighbor_embeddings_folder,
            chip_transforms=chip_transforms,
            get_osm_strlabels = get_osm_strlabels,
            get_osm_ohecount = get_osm_ohecount,
            get_osm_ohearea = get_osm_ohearea,
            get_osm_ohelength = get_osm_ohelength,
            get_esawc_proportions = get_esawc_proportions,
            get_chip_id = get_chip_id,
            max_items = max_items,
            embeddings_normalization = embeddings_normalization,
            osmvector_normalization = osmvector_normalization,
            osm_codeset = osm_codeset,
            multilabel_threshold_osm_ohecount = multilabel_threshold_osm_ohecount,
            multilabel_threshold_osm_ohearea = multilabel_threshold_osm_ohearea,
            split="val",
        )

        self.test_dataset = ChipMultilabelDataset(
            metadata_file = metadata_file,
            chips_folder = chips_folder,
            embeddings_folder = embeddings_folder,
            patch_embeddings_folder = patch_embeddings_folder,
            neighbor_embeddings_folder = neighbor_embeddings_folder,
            chip_transforms = chip_transforms,
            get_osm_strlabels = get_osm_strlabels,
            get_osm_ohecount = get_osm_ohecount,
            get_osm_ohearea = get_osm_ohearea,
            get_osm_ohelength = get_osm_ohelength,
            get_esawc_proportions = get_esawc_proportions,
            get_chip_id = get_chip_id,
            max_items = max_items,
            embeddings_normalization = embeddings_normalization,
            osmvector_normalization = osmvector_normalization,
            osm_codeset = osm_codeset,
            multilabel_threshold_osm_ohecount = multilabel_threshold_osm_ohecount,
            multilabel_threshold_osm_ohearea = multilabel_threshold_osm_ohearea,
            split="test",
        )

        # setup normalizer object
        self.normalizer = OSMandEmbeddingsNormalizer(self)
        self.train_dataset.set_osm_and_embeddings_normalizer(self.normalizer)
        self.test_dataset.set_osm_and_embeddings_normalizer(self.normalizer)
        self.val_dataset.set_osm_and_embeddings_normalizer(self.normalizer)

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