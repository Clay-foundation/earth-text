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
        for ax, i in subplots(4, n_cols=4, usizex=4):
            if i==0:
                rgbimg = np.transpose(self.img[:3],[1,2,0]).copy()
                rgbimg /= 2000
                rgbimg[rgbimg>1]=1
                plt.imshow(rgbimg)
            else:
                _x = self.img[i-3]
                a,b = np.percentile(_x, [1,99])
                plt.imshow(_x, vmin=a, vmax=b)
                plt.colorbar()
                if i==0:
                    plt.title(self.chip_id)
                


    def get_onehot(self):
        """
        returns
        - a dataframe with idexes 'area', 'length' and 'count' of each tag present in this chip
          after filtering according to kvopen and kvclose, with the tags encoded by their
          number code and columns ordered and filled in with zeros as a onehot encoding
        - a list of the strings representing the tags present
        """

        flatten = lambda x: [i for j in x for i in j]
        t2str = lambda t: list(np.unique(flatten( [ [f"{k}={v}", f"{k}=*"] for k,v in t.items()])))
        
        oosm = self.osm.copy()
        del (oosm['geometry'])
        oosm = oosm[[i=='way' for i in oosm.kind.values]].copy()
        oosm['is_closed'] = [i>0 for i in oosm['area'].values]
        oosm['is_open'] = [i==0 for i in oosm['area'].values]
        
        oosm_open = oosm[oosm.is_open].copy()
        oosm_closed = oosm[oosm.is_closed].copy()
        
        # filter tags
        oosm_open['tags'] = [kvopen.filter_keyvals(t) for t in oosm_open.tags.values]
        oosm_closed['tags'] = [kvclosed.filter_keyvals(t) for t in oosm_closed.tags.values]
        
        # remove osm objects which ended up with no tags
        oosm_open   = oosm_open[[len(t)>0 for t in oosm_open.tags.values]]
        oosm_closed = oosm_closed[[len(t)>0 for t in oosm_closed.tags.values]]
        
        # merge all
        oosm = pd.concat([oosm_closed, oosm_open])
        
        # generate string keyvals for multilabel
        oosm['stags'] = [ t2str(t) for t in oosm['tags']]
        
        # get the multilabel code for each surviving tag
        oosm['ctags'] = [[kvmerged.keyvals_codes[si] for si in s] for s in oosm.stags.values]
        
        # counts, areas and lengths for each tag
        ocounts = np.zeros(len(kvmerged.keyvals_codes)).astype(int)
        oareas = np.zeros(len(kvmerged.keyvals_codes))
        olengths = np.zeros(len(kvmerged.keyvals_codes))
        for _,oi in oosm.iterrows():
            for c in oi.ctags:
                ocounts[c] += 1
                oareas[c] += oi.area
                olengths[c] += oi.length
        
        # check we count the ones with area plus the ones with length
        assert np.all( (ocounts==0) == ( (oareas==0) & (olengths==0)))
        
        # make a list of keyval strings for which there is a count > 0
        ostrs = [kvmerged.inverse_codes[i] for i in range(len(ocounts)) if ocounts[i]!=0]
        
        # assemble a dataframe
        oor = pd.DataFrame([oareas, olengths, ocounts], index=['area', 'length', 'count'])

        return oor, ostrs

    def xget_onehot(self):
        """
        returns
        - a dataframe with idexes 'area', 'length' and 'count' of each tag present in this chip
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