import numpy as np
import pandas as pd
from loguru import logger
import torch
from torch.utils.data import Dataset
from progressbar import progressbar as pbar
from earthtext.io import io
from earthtext.osm import multilabel
import matplotlib.pyplot as plt
import geopandas as gpd
import os
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


norm_names = {
    'mean_stdev_norm': 'ok'
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
        neighbor_embeddings_folder: str = None,
        patch_embeddings_folder: str = None,
        chip_transforms = None,
        get_osm_ohecount = False,
        get_osm_ohearea = False,
        get_osm_ohelength = False,
        get_osm_strlabels = False,
        get_esawc_proportions = False,
        get_chip_id = False,
        multilabel_threshold_osm_ohecount = None,
        multilabel_threshold_osm_ohearea = None,
        max_items = None,
        embeddings_normalization = True,
        osmvector_normalization = False,
        osm_codeset = 'sentinel2',
        cache_size = -1
    ):

        """
        get_osm_strlabels: return also the string description of the multilabels present
        multilabel_threshold_osm_ohecount: returns a 0/1 ohe vector with 1 if the ohe count is > min_ohecount
                       if None, it will return the ohe count
        """
        if sum([multilabel_threshold_osm_ohecount is None, multilabel_threshold_osm_ohearea is None])!=1:
            raise ValueError("must specify exactly one of 'multilabel_threshold_osm_ohearea' or 'multilabel_threshold_osm_ohecount'")

        self.split = split
        self.chips_folder = chips_folder
        self.chip_transforms = chip_transforms
        self.embeddings_folder = embeddings_folder
        self.neighbor_embeddings_folder = neighbor_embeddings_folder
        self.patch_embeddings_folder = patch_embeddings_folder
        self.get_esawc_proportions = get_esawc_proportions
        self.metadata_file = metadata_file
        self.get_chip_id = get_chip_id
        self.get_osm_strlabels = get_osm_strlabels
        self.get_osm_ohearea = get_osm_ohearea
        self.get_osm_ohecount = get_osm_ohecount
        self.get_osm_ohelength = get_osm_ohelength
        self.max_items = max_items
        self.embeddings_normalization = embeddings_normalization
        self.osmvector_normalization = osmvector_normalization
        self.metadata = io.read_multilabel_metadata(metadata_file)
        self.metadata = self.metadata[self.metadata['split']==split]
        self.multilabel_threshold_osm_ohecount = multilabel_threshold_osm_ohecount
        self.multilabel_threshold_osm_ohearea = multilabel_threshold_osm_ohearea
        self.osm_codeset = osm_codeset

        if 'embeddings' in self.metadata.columns:
            if self.embeddings_folder is not None:
                raise ValueError("cannot set 'embeddings_folder' since metadata already has an 'embeddings' column")
            self.metadata_has_embeddings = True
            logger.info("using embeddings found in metadata file")
        else:
            self.metadata_has_embeddings = False


        nitems = len(self.metadata)
        # keep only the items for which there is actually a chip image file
        if chips_folder is not None:
            logger.info(f"checking chip files for {split} split")
            chips_exists = [io.check_chip_exists(chips_folder, embeddings_folder, patch_embeddings_folder, i['col'], i['row']) \
                                for _, i in pbar(self.metadata.iterrows(), max_value=len(self.metadata))]
            self.metadata = self.metadata[chips_exists]

        logger.info(f"read {split} split with {len(self.metadata)} chip files (out of {nitems})")

        # Remove chip IDs with no associated neighbors .npy files
        if self.neighbor_embeddings_folder is not None:
            logger.info(f"removing chip IDs with no associated neighbors .npy files")
            folder = self.neighbor_embeddings_folder
            files = pd.Series([f.removesuffix('.npy') for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            self.metadata = self.metadata[self.metadata['original_chip_id'].isin(files)]

        if max_items is not None:
            self.metadata = self.metadata.iloc[np.random.permutation(len(self.metadata))[:max_items]]
            logger.info(f"limitting to {max_items} items, randomly selected")
        logger.info(f"max cache size is {cache_size}")
        self.cache_size = cache_size
        self.cache = {}

        # so that we can disable chip loading. For instance, we would want to disable it when
        # training with embeddings, and enable it for visualization
        self.disable_chip_loading = False

        self.normalizer = None

    def prepare_data(self):
        """This is an optional preprocessing step to be defined in each dataloader.
        It will be called by the Pytorch Lighting Data Module.
        All the preprocessing steps such as normalization or transformations
        might be write in this method when you override it.
        """

    def set_osm_and_embeddings_normalizer(self, normalizer):
        self.normalizer = normalizer

    def __len__(self):
        return len(self.metadata)


    def __repr__(self):
        return f"{self.__class__.__name__} {self.split} split with {len(self)} items, in cache {len(self.cache)} items"


    def __getitem__(self, idx):

        if idx in self.cache.keys():
            return self.cache[idx]

        r = {}        

        item = self.metadata.iloc[idx]
        if self.multilabel_threshold_osm_ohecount is not None:
            multilabel = item.onehot_count.astype(int)
            multilabel = (multilabel >= self.multilabel_threshold_osm_ohecount).astype(int)

        if self.multilabel_threshold_osm_ohearea is not None:
            # either area or a bit less than squared length
            min_ohe_length = np.sqrt(self.multilabel_threshold_osm_ohearea)*4 / 1.5
            multilabel = (item.onehot_area > self.multilabel_threshold_osm_ohearea) | (item.onehot_length > min_ohe_length)

        r['multilabel'] = torch.tensor(multilabel).type(torch.int8)

        if self.get_chip_id:
            r['chip_id'] = item.name

        if self.chips_folder is not None and not self.disable_chip_loading:
            chip = io.read_chip(self.chips_folder, item['col'], item['row'])
            if self.chip_transforms is not None:
                chip = self.chip_transforms(chip)
            r['chip'] = chip

        if self.neighbor_embeddings_folder is not None:
            r['embedding'] = io.read_neighbor_embeddings(embeddings_folder=self.neighbor_embeddings_folder,
                                                         original_chip_id=item['original_chip_id'])
            if self.embeddings_normalization:
                r['embedding'] = self.normalizer.normalize_embeddings(r['embedding'])
        elif self.embeddings_folder is not None:
            r['embedding'] = io.read_embedding(self.embeddings_folder,  item['col'], item['row'])
            if self.embeddings_normalization:
                r['embedding'] = self.normalizer.normalize_embeddings(r['embedding'])
        elif self.metadata_has_embeddings:
            r['embedding'] = item['embeddings'].copy()
            if self.embeddings_normalization:
                r['embedding'] = self.normalizer.normalize_embeddings(r['embedding'])


        if self.patch_embeddings_folder is not None:
            r['patch_embedding'] = io.read_patch_embedding(self.patch_embeddings_folder,  item['col'], item['row'])

        if self.get_osm_strlabels:
            r['osm_strlabels'] = " ".join(item.string_labels)

        if self.get_osm_ohearea:
            r['osm_ohearea'] = np.r_[item.onehot_area]
            if self.osmvector_normalization:
                r['osm_ohearea'] = self.normalizer.normalize_osm_vector_area(r['osm_ohearea'])

        if self.get_osm_ohecount:
            r['osm_ohecount'] = np.r_[item.onehot_count]
            if self.osmvector_normalization:
                r['osm_ohecount'] = self.normalizer.normalize_osm_vector_count(r['osm_ohecount'])

        if self.get_osm_ohelength:
            r['osm_ohelength'] = np.r_[item.onehot_length]
            if self.osmvector_normalization:
                r['osm_ohelength'] = self.normalizer.normalize_osm_vector_length(r['osm_ohelength'])
                
        if self.get_esawc_proportions:
            r['esawc_proportions'] = str(item.esawc_proportions)

        # store in cache
        if self.cache_size == -1 or len(self.cache) < self.cache_size:
            self.cache[idx] = r

        return r

    def reset_cache(self):
        self.cache = {}

    def plot_chip_with_tags(self, idx, osm_tags):
        
        if not self.get_chip_id:
            raise ValueError("must set 'get_chip_id' to true in this dataset to plot chips")
        
        if sum(["=" in i for i in osm_tags])!=len(osm_tags):
            raise ValueError("must use '=' in all tags")
        
        # load osm objects
        osm_folder = "/".join(self.embeddings_folder.split("/")[:-1])+"/osm"
        chip_record = self.metadata.iloc[idx]
        chip_id = chip_record.name
        z = gpd.read_parquet(f"{osm_folder}/{chip_id}.parquet")

        # filter tags
        asterix_tags = [i.split("=")[0] for i in osm_tags if i.endswith("=*")]
        regular_tags = {i.split("=")[0]:i.split("=")[1] for i in osm_tags if not i.endswith("=*")}
        
        def hastags(tags, asterix_tags, regular_tags):
            if len(set(tags.keys()).intersection(asterix_tags))>0:
                return True
        
            for k,v in regular_tags.items():
                if k in tags.keys() and v == tags[k]:
                    return True
            return False
        
        zf = z[[hastags(eval(t), asterix_tags, regular_tags) for t in z.tags.values]]
        
        # get chip
        item = self[idx]
        c = item['chip'].numpy()[:3]
        c = np.transpose(c, [1,2,0])
        a,b = np.percentile(c, [5,99])
        c = c *1.0 / b
        c[c>1]=1
        
        # plot chip and geometries
        xy = np.r_[[chip_record.geometry.exterior.xy]][0]
        minlon, minlat = xy.min(axis=1)
        maxlon, maxlat = xy.max(axis=1)
        plt.imshow(c, extent=(minlon, maxlon, minlat, maxlat), alpha=.7)
        
        def plot_geom(geom):
            if 'geoms' in dir(geom):
                for gi in geom.geoms:
                    plot_geom(gi)
            elif 'exterior' in dir(geom):
                plt.plot(*geom.exterior.xy, color="red", alpha=1, lw=2)
            elif 'xy' in dir(geom):
                plt.plot(*geom.xy, color="red", alpha=1, lw=2)
        
        for xgeom in zf.geometry:
            k = plot_geom(xgeom)
        
        plt.axis("off");
        
