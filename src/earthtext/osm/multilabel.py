import rasterio
import xarray as xr
import os
from rlxutils import subplots
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from collections import OrderedDict
import pandas as pd
from . import osm

class ImageOSMData:

    def __init__(self, imgs_folder, osmobjs_folder, master_index):
        self.imgs_folder = imgs_folder
        self.osmobjs_folder = osmobjs_folder
        self.master_index = master_index

    def init_index(self):
        self.m = gpd.read_file(self.master_index)
        self.m.index = [osm.get_region_hash(gi) for gi in self.m.geometry.values]
        return self

    def sample_chip(self):
        chip_id = self.m.index[np.random.randint(len(self.m))]
        return ImageOSMChip(self, chip_id)
    
class OSMKeyValueCodes:

    def __init__(self, kind):

        if not kind in ['openways', 'closedways']:
            raise ValueError(f"invalid kind '{kind}', kind must be 'openways' or 'closedways'")

        if kind=='closedways':
            raw_osm_keyvals = {}
            raw_osm_keyvals['landuse']  = ['residential', 'grass', 'farmland', 'meadow', 'commercial', 'orchard', 'vineyard', 'industrial', 
                                                   'retail', 'farmyard', 'forest', 'military', 'farm', 'cemetery', 'brownfield', 'quarry', 'greenfield']
            raw_osm_keyvals['natural']  = ['water', 'wood', 'scrub', 'sand', 'grassland', 'wetland', 'bare_rock', 'coastline', 'heath', 'valley', 
                                                  'desert', 'cliff', 'scree', 'beach', 'mountain_range', 'mud', 'bay']
            raw_osm_keyvals['leisure']  = ['pitch', 'park', 'garden', 'nature_reserve', 'sports_centre', 'golf_course', 'track', 'schoolyard', 'stadium']
            raw_osm_keyvals['water']    = ['pond', 'reservoir', 'lake', 'river', 'canal', 'wastewater', 'stream', 'ditch', 'stream;river']
            raw_osm_keyvals['sport']    = ['baseball', 'soccer', 'american_football', 'running', 'equestrian', 'athletics', 'motor', 'multi']
            raw_osm_keyvals['building'] = ['house', 'residential', 'apartments', 'industrial', 'school', 'warehouse']
            raw_osm_keyvals['surface']  = ['asphalt', 'concrete', 'paved', 'gravel', 'sand', 'grass']
            raw_osm_keyvals['crop']     = ['grape', 'field_cropland', 'WINE GRAPES', 'native_pasture']
            raw_osm_keyvals['power']    = ['generator', 'substation', 'plant']
            raw_osm_keyvals['parking']  = ['surface', 'multi-storey']
            raw_osm_keyvals['highway']  = ['track']
            raw_osm_keyvals['waterway'] = ['dam']
            raw_osm_keyvals['amenity']  = ['parking']
            
        elif kind == 'openways':
            raw_osm_keyvals = {}
            raw_osm_keyvals['highway']  = ['motorway', 'residential', 'track']
            raw_osm_keyvals['waterway'] = ['stream', 'river']

        self.raw_osm_keyvals = raw_osm_keyvals
        self.init_codes()

    def init_codes(self):
        # maps to remove duplicated and standardize
        self.osm_keyvals = {k: list(set([self.map_keyval(k,vi)[1] for vi in v])) for k,v in self.raw_osm_keyvals.items()}
        
        # append key=* to account for single keys
        for k in self.osm_keyvals.keys():
            self.osm_keyvals[k].append("*")
                
        # assigned a code to each keyvalue pair
        self.keyvals_codes = {}
        code = 0
        for k in sorted(list(self.osm_keyvals.keys())):
            for vi in sorted(self.osm_keyvals[k]):
                self.keyvals_codes[f"{k}={vi}"] = code
                code += 1

        self.inverse_codes = {v:k for k,v in self.keyvals_codes.items()}
    
    def map_keyval(self, key, val):
        """
        maps a key, val pair to remove duplicates, standardize, etc.
        
        keyvals: a dict with osm key values

        returns: a (key,value) tuple with the mapped keyval, or (None, None) 
        """
        k,v = key,val
        
        if k=='crop' and v=='WINE GRAPES':
            k, v = 'crop', 'grape'
        if k=='water' and v=='stream;river':
            k, v = 'water', 'stream'

        return k, v

    def filter_keyvals(self, keyvals):
        """
        filters the key/value pairs to only those considered in the constructor
        after mapping to stardardize them
        """        
        
        # cleans and standardizes keyvals
        keyvals = {kk:vv for kk,vv in [self.map_keyval(k,v) for k,v in keyvals.items()]}
        
        # keep only keyvals as specified in __init__
        keyvals = {k:v for k,v in keyvals.items() if k in self.osm_keyvals.keys() and v in self.osm_keyvals[k]}

        return keyvals
        
    def get_codes(self, keyvals):
        """
        returns a list of codes, one for each key/value pair in the input dictionary
        it returns only the codes for the key/value pairs considered in the constructor
        after mapping to stardardize them
        
        keyvals: a dictionary
        """

        keyvals = self.filter_keyvals(keyvals)
        
        r = []
                     
        for k,v in keyvals.items():
            k,v = self.map_keyval(k,v)
            kvstring = f'{k}={v}'
            if kvstring in self.keyvals_codes.keys():
                code  = self.keyvals_codes[kvstring]
                r.append(code)
        return r  


class OSMMergedKeyValueCodes:

    def __init__(self, kv1, kv2):
        """
        kv1, kv2: two objects of type OSMKeyValueCodes
        """
        self.kv1 = kv1
        self.kv2 = kv2

        # merge the two dicts of codes by looping over kv2
        kv = kv1.keyvals_codes.copy()
        next_code = np.max(list(kv.values())) + 1
        for k,v in kv2.keyvals_codes.items():
            if k in kv.keys():
                continue
            kv[k] = next_code
            next_code += 1

        self.keyvals_codes = kv
        self.inverse_codes = {v:k for k,v in self.keyvals_codes.items()}

    def merge_codes(self, codes1, codes2):
        """
        codes1: list of codes (ints) from OSMKeyValueCodes object 1
        codes2: list of codes (ints) from OSMKeyValueCodes object 2
        """

        r = [self.keyvals_codes[self.kv1.inverse_codes[i]] for i in codes1] + \
            [self.keyvals_codes[self.kv2.inverse_codes[i]] for i in codes2]
        return r

kvopen   = OSMKeyValueCodes(kind='openways')
kvclosed = OSMKeyValueCodes(kind='closedways')
kvmerged = OSMMergedKeyValueCodes(kvclosed, kvopen)
max_code = np.max(list(kvmerged.keyvals_codes.values()))

class ImageOSMChip:

    def __init__(self, iosmdata, chip_id):

        self.iosmdata = iosmdata
        self.chip_id = chip_id
        
    def read_osm(self, min_area=0):
        """
        reads the osm objects for this chip id, cleaning up key/values
        """
        if not self.chip_id in self.iosmdata.m.index:
            raise ValueError(f"chip {self.chip_id} not indexed")
            
        try:
            z = gpd.read_parquet(f"{self.iosmdata.osmobjs_folder}/{self.chip_id}.parquet")
            z['tags'] = [eval(i) if isinstance(i, str) else i for i in z.tags]
            z['tags'] = [osm.clean_tags(t) for t in z.tags]
            z = z[[len(t)>0 for t in z['tags']]]
            z = z[z['area']>=min_area]
    
            if len(z)==0:
                self.osm = None
            else:
                self.osm = z.copy()
        except:
            self.osm = None

        return self

    def read_img(self):
        if not self.chip_id in self.iosmdata.m.index:
            raise ValueError(f"chip {self.chip_id} not indexed")
            
        mi = self.iosmdata.m.loc[self.chip_id]
        fname = f"{self.iosmdata.imgs_folder}/worldcover_california_chip_{mi['col']}_{mi['row']}.tif"
        
        with xr.open_dataarray(fname) as z:
            self.img = z.data.copy()

        return self
    
    def show(self):
        for ax, i in subplots(len(self.img), n_cols=6, usizex=4):
            _x = self.img[i]
            a,b = np.percentile(_x, [1,99])
            plt.imshow(_x, vmin=a, vmax=b)
            plt.colorbar()
            if i==0:
                plt.title(self.chip_id)
                
    def get_multilabel_keyvals(self, remove_duplicates=True):
        codes = self.get_multilabel_codes(remove_duplicates=remove_duplicates)
        return [kvmerged.inverse_codes[code] for code in codes]
    
    def get_multilabel_codes(self, remove_duplicates=True):
        # get tags for open and closed ways
        tags_openways   = [o.tags for _,o in self.osm.iterrows() if o.kind=='way' and o.area==0]
        tags_closedways = [o.tags for _,o in self.osm.iterrows() if o.kind=='way' and o.area>0]
    
        # gets the codes for open and closed tags
        open_codes = [i for t in tags_openways for i in kvopen.get_codes(t)]
        closed_codes = [i for t in tags_closedways for i in kvclosed.get_codes(t)]
    
        # merges them
        codes = kvmerged.merge_codes(closed_codes, open_codes)
    
        # if remove duplicates 
        if remove_duplicates:
            codes = list(set(codes))
    
        # sorts them
        codes = sorted(codes)
    
        return codes

    def get_multilabel_onehot(self):
        codes = self.get_multilabel_codes(remove_duplicates=True)
        ohcodes = np.zeros(max_code+1).astype(int)
        ohcodes[codes] = 1
        return ohcodes

    def get_onehot(self):
        """
        returns
        - a dataframe with 'area', 'length' and 'count' of each tag present in this chip
          after filtering according to kvopen and kvclose, with the tags encoded by their
          number code and columns ordered and filled in with zeros as a onehot encoding
        - a list of the strings representing the tags present
        """

        osm = self.osm.copy()
        osm = osm[[i=='way' for i in osm.kind.values]].copy()
        osm['is_closed'] = [i>0 for i in osm['area'].values]
        osm['is_open'] = [i==0 for i in osm['area'].values]

        def add_stats(keyval_labels, key, val, area, length):
            label = f"{key}={val}"
            if not label in keyval_labels.keys():
                keyval_labels[label] = {'area':0, 'length':0, 'count':0}
        
            keyval_labels[label]['area'] += area
            keyval_labels[label]['length'] += length
            keyval_labels[label]['count'] += 1
                
        
        # aggregate area, length and count for each osm keyval
        keyval_labels = {}
        for _,o in osm.iterrows():
            if o.is_closed:
                tags = kvclosed.filter_keyvals(o.tags)
            else:
                tags = kvopen.filter_keyvals(o.tags)
            for k,v in tags.items():
                add_stats(keyval_labels, k,v, o['area'], o['length'])
                add_stats(keyval_labels, k,'*', o['area'], o['length'])
        
        # dataframe with area, length and count per osm keyval
        keyval_labels = pd.DataFrame(keyval_labels)
        keyval_strs = sorted(keyval_labels.columns)
        
        # dataframe with codes instead of osm keyval
        keyval_onehot = keyval_labels.copy()
        keyval_onehot.columns = [kvmerged.keyvals_codes[c] for c in keyval_onehot.columns]
        
        # fillin non present codes with zeros
        for code in range(np.max(list(kvmerged.keyvals_codes.values()))+1):
            if not code in keyval_onehot.columns:
                keyval_onehot[code] = [0,0,0]
        
        # sort columns
        keyval_onehot = keyval_onehot[sorted(keyval_onehot.columns)]

        return keyval_onehot,keyval_strs