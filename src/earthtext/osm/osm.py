import geopandas as gpd
import shapely as sh
from pyproj import CRS
epsg4326 = CRS.from_epsg(4326)
import pandas as pd
from progressbar import progressbar as pbar
import overpy
api = overpy.Overpass()
import numpy as np
import matplotlib.pyplot as plt
import hashlib

from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection

def subplots(elems, n_cols=None, usizex=3, usizey=3, **ax_kwargs ):
    """
    generates grid with a subplot for each elem in elems. for instance:

       for ax,i in subplots(2):
           if i==0:
              ax.scatter(....)
           if i==1:
              ax.plot(...)


    elems: an iterable, or an integer
    n_cols: the number of columns for the grid. If none it is set to min(num elems, 15)
    usizex, usizey: size of each subplot
    ax_kwargs: args to pass to axis creation (such as projection, etc.)
    """

    if type(elems)==int:
        elems = np.arange(elems)

    n_elems = len(elems)

    if n_cols is None:
        n_cols = n_elems if n_elems<=15 else 15

    n_rows = n_elems//n_cols + int(n_elems%n_cols!=0)
    fig = plt.figure(figsize=(n_cols*usizex, n_rows*usizey))

    for i in range(n_elems):
        ax = fig.add_subplot(n_rows, n_cols, i+1, **ax_kwargs)
        yield ax, elems[i]

    plt.tight_layout()

def katana(geometry, threshold, count=0, random_variance=0.1):
    """
    splits a polygon recursively into rectangles
    geometry: the geometry to split
    threshold: approximate size of rectangles
    random_variance: 0  - try to make all rectangles of the same size
                     >0 - the greater the number, the more different the rectangle sizes
                     values between 0 and 1 seem more useful
                     
    returns: a list of Polygon or MultyPolygon objects
    """
    
    
    """Split a Polygon into two parts across it's shortest dimension"""
    assert random_variance>=0

    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    
    random_factor = 2*(1+(np.random.random()-0.5)*random_variance*2)
    
    if max(width, height) <= threshold or count == 250:
        # either the polygon is smaller than the threshold, or the maximum
        # number of recursions has been reached
        return [geometry]
    if height >= width:
        # split left to right
        a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/random_factor)
        b = box(bounds[0], bounds[1]+height/random_factor, bounds[2], bounds[3])
    else:
        # split top to bottom
        a = box(bounds[0], bounds[1], bounds[0]+width/random_factor, bounds[3])
        b = box(bounds[0]+width/random_factor, bounds[1], bounds[2], bounds[3])
    result = []
    for d in (a, b,):
        c = geometry.intersection(d)
        if not isinstance(c, GeometryCollection):
            c = [c]
        for e in c:
            if isinstance(e, (Polygon)):
                result.extend(katana(e, threshold, count+1, random_variance))
            if isinstance(e, (MultiPolygon)):
                for p in e.geoms:
                    result.extend(katana(p, threshold, count+1, random_variance))
    if count > 0:
        return result
    # convert multipart into singlepart
    final_result = []
    for g in result:
        if isinstance(g, MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return final_result

class OSMAOI:
    
    def __init__(self, aoi, katana_threshold=1, timeout=60, key='natural'):
        self.aoi = aoi
        self.katana_threshold = katana_threshold
        self.timeout = int(timeout)
        self.key = key
        
        self.geoms = katana(self.aoi, katana_threshold)
        self.geom = sh.geometry.GeometryCollection(self.geoms)
        self.nodes = {}#self.gethash(g):[] for g in self.geoms}
        self.ways  = {}#self.gethash(g):[] for g in self.geoms}
        self.rels  = {}#self.gethash(g):[] for g in self.geoms}
        
    def gethash(self, region):
        """
        region: a shapely geometry
        returns a hash string for region using its coordinates
        """
        s = str(np.r_[region.envelope.boundary.coords].round(5))
        k = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**15
        k = str(hex(k))[2:].zfill(13)
        return k        
        
    def getobjs(self, objtype):
        if not objtype in ['node', 'way', 'rel']:
            raise ValueError(f"invalid obj type {objtype}")
            
        area_coords = lambda x: " ".join([f"{lat:.3f} {lon:.3f}" for lon, lat in list(x.boundary.coords)])            
        
        h = self.gethash
        store = self.nodes if objtype == 'node' else self.ways if objtype == 'ways' else self.rels
        
        print (f"retrieving {len(self.geoms) - len(store)} missing objects of type {objtype} out of {len(self.geoms)}", flush=True)
        for geom in pbar(self.geoms):
            if h(geom) in store.keys():
                continue
                
            while True:
                tries = 0
                try:
                    r = api.query(f"""
                        [timeout:{self.timeout}];
                        ({objtype}["{self.key}"](poly: "{area_coords(geom)}");
                        );
                        out;
                        """)
                    break
                except Exception as e:
                    print(f"retrying {h(geom)}", e, flush=True)
                    tries += 1
                    if tries>4:
                        print (f"could not get {h(geom)}, giving up")
                        break

            store[h(geom)] = r

