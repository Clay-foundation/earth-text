import geopandas as gpd
import shapely as sh
from shapely.geometry import Point
from pyproj import CRS
epsg4326 = CRS.from_epsg(4326)
epsg_california = CRS.from_epsg(26946)
import pandas as pd
from progressbar import progressbar as pbar
import progressbar
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import osmium
import sys
import os
from joblib import Parallel

from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection

class mParallel(Parallel):
    """
    substitutes joblib.Parallel with richer verbose progress information
    """
    def _print(self, msg):
        if self.verbose > 10:
            fmsg = '[%s]: %s' % (self, msg)
            sys.stdout.write('\r ' + fmsg)
            sys.stdout.flush()


#def remove_ignored_tags(t):
#    ignore_tags = ['created_by', 'add:', 'gnis:', 'gtfs_id', 'tiger:', 'name', 'source', 'nhd:', 'wikipedia', 'ref']
#    return {k:v for k,v in t.items() if not k in ignore_tags and sum([k.startswith(ti) for ti in ignore_tags if ti[-1]==':'])==0}    

def get_region_hash(region):
    """
    region: a shapely geometry
    returns a hash string for region using its coordinates
    """
    s = str(np.r_[region.envelope.boundary.coords].round(5))
    k = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k

tags2str = lambda x: "\n<br>\n".join([f"{k}: {v}" for k,v in x.items()])

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

def clean_tags(tags):
    # cleaup tags and remove objects with no tags
    ignore_tags = ['created_by', 'addr:', 'gnis:', 'gtfs_id', 'tiger:', 'name', 'source', 'nhd:', 'wikipedia', 'ref', 'attribution',
                  'note', 'protection_title', 'wikidata', 'FMMP_modified', 'FMMP_reviewed', 'ref:', 'plant:', 'access',
                  'contact', 'acres', 'ele', 'oldref', 'oneway', 'lanes:', 'boundary', 'border_type', 'admin_level', 'fixme',
                  'altname', 'lanes', 'maxspeed', 'maxspeed:', 'operator:', 'operator', 'COUNTY_NAM', 'source:', 'website', 'old_ref', 'name_1',
                  'alt_name', 'name:', 'handicapped_accessible', 'old_railway_operator', 'caltrans:', 'ALAND', 'AREAID',
                  'AWATER', 'MTFCC', 'latitude', 'longitude', 'military', 'source_ref', 'payment:', 'ohv', 'lot_description','lot_type',
                  'barrier', 'access:', 'horse', 'tracktype', 'layer', 'cables', 'volcano:', 'intermitent', 'foot', 'bicycle', 'motor_vehicle',
                  'seasonal', 'description', 'proposeinternet_accessd:', 'bakersfield:', 'official_name', 'official_name_1', 'destination', 'junction:',
                  'owner', 'gauge', 'NHS', 'abandoned:', 'opening_hours', 'meadlsc_description', 'distance', 'FIXME', 'toilets:',
                  'hgv:', 'capacity', 'capacity:', 'fee', 'intermittent', 'ID', 'county', 'csp:', 'traffic_signals', 'traffic_signals:',
                  'lacounty:', 'start_date', 'height', 'width', 'direction', 'voltage', 'monitoring:', 'man_made', 'hgv', 'kern:',
                  'turn:', 'brand:', 'brand', 'wheelchair', 'cuisine', 'not:', 'stop', 'short_name', 'demolished:', 'clothes',
                  'network', 'network:',  'traffic_calming', 'phone', 'denomination', 'religion', 'healthcare', 'healthcare:',
                  'old_name', 'area', 'contact', 'contact:', 'addr2:', 'advertising', 'check_date', 'designation', 'fuel:',
                  'internet_access', 'reciclying_Type', 'atv', 'height_ft', 'width_ft', 'width:', 'old_name:', 'note_1',
                  'destination:', 'bridge:name', 'bridge_ref', 'source_ref:', 'mono:', 'noname', 'cycleway:', 'disused:', 'generator:',
                  'circuits', 'covered', 'end_date', 'frequency', 'electrified', 'informal', 'historic', 'lit', 'milepost:', 'noexit',
                  'toll', 'maxwidth', 'wires', 'tracktype', 'mtb', 'COUNTYFP', 'Tiger:', 'wikimedia_commons', 'winter_service',
                  'STATEFP', 'image', 'handicap', 'Open_Date', 'Parks_ID', 'Park_No', 'takeaway', 'url', 'unsigned_ref', 'tract', 
                  'trail_visibility', 'surface:note', 'street:', 'step_count', 'smoking', 'abandoned', 'sidewalk', 'sidewalk:', 'roof:',
                  'reviewed', 'ramp:', 'recycling_type', 'recycling:', 'postal_code', 'piste:', 'passenger_lines', 'proposed' ,'par', 
                  'notes', 'note_2', 'internet_access:', 'gns:', 'ANSICODE', 'architect', 'artist_name', 'artwork_type', 'city_served',
                  'faa', 'fax', 'hoops', 'manufacturer', 'FIXME:', 'atribution', 'alt_name:', 'bridge:', 'check_date:', 'comm_code',
                  'datum:', 'dt_add', 'fields_id', 'fields', 'maxlength:', 'name_2', 'no_prmt_si', 'no_prmt', 'note:', 'no_site',
                  'oper_add', 'old_ref:', 'scvwd:', 'mtb:', 'lot_no', 'hov:', 'date_on', 'dog', 'golf_cart', 'closest_town', 'beds', 'bench',
                   'change:', 'handrail', 'opening_date', 'outdoor_seating', 'orientation', 'population', 'placement', 'sangis:', 'src:',
                  'unisex', 'bak:', 'description2', 'drive_through', 'drink:', 'email', 'facility_name', 'is_in', 'information',
                   'maxheight', 'maxweight', 'ownership', 'railway:', 'was:', 'road_marking', 'restriction', 'route', 'screen',
                   'psv', 'private', 'odbl', 'odbl:', 'nudism', 'year_planted', 'tactile_paving', 'vehicle:', 'verified:', 'utility_wires',
                   'opening_hours:', 'is_in:', 'length', 'area:', 'bicycle:', 'colour', 'capture', 'dt_mant', 'heritage:', 'len',
                   'otis_id', 'nist:', 'primary_use', 'property_number', 'protect_class', 'shape_area', 'assmntdist', 'cost', 'name_alt', 'region',
                   'shape_le_1', 'shape_len', 'abutters', 'access_aisle', 'branch', 'district', 'grades', 'managed', 'male', 'store_number',
                   'subject', 'subject:', 'beauty', 'atm', 'animal', 'changing_table', 'cyclerating', 'fitness_station', 'lcn', 'membership', 
                   'min_age', 'mofa', 'maxweight:', 'max_age', 'motor_vehicle:', 'placement:', 'sac_scale', 'route_ref', 'zoning',
                   '_shape_area_', '_acres_', '_shape_leng_', 'agncy_name', 'agncy_id', 'agncy_lev', 'agncy_type', 'agncy_web',
                   'des_tp', 'delivery', 'gap_sts', 'government', 'label_name', 'shape_star', 'shapestare', 'shape_stle', 'unit_id', 
                   'unit_name', 'maxstay', 'elevation', 'self_service','zone', 'upload_version', 'sbc_apn', 'sbc_id', 'sbc_parcel', 'quantity',
                   'phases', 'other_use', 'objectid', 'nrhp:', 'maintained', 'inscription', 'inscription:', 'incline', 'collection_times',
                   'climbing:', 'website:', 'unit', 'service_times', 'shapestlen', 'rooms', 'ramp', 'proposed:', 'parking:', 'fence_type',
                   'departures_board', 'crossing:', 'comment', 'apn', 'x_coordinate', 'y_coordinate', 'gtfs_location_type', 'gtfs_stop_code', 
                   'flag', 'flag:', 'socket', 'socket:', 'hfcs', 'image_1', 'image', 'loc_name', 'beacon:', 'redwood_city_ca:',
                   'nhd-shp:', 'adot_name', '3dr:', '4wd_only', '_address_', '_area', '_bldg_no_', '_desc__', '_fmssid_', '_globalid_',
                   '_id_', '_name_', '_lcsid_', '_tnode__', '_type_', 'abbr_name', 'addr','address', 'address:', 'type', 'building:', 
                   'paloalto_ca:', 'massgis:', 'construction:', 
                ]

    ignore_tags = [i.lower() for i in ignore_tags]
    
    # comments
    #  - may want to keep lanes info in the future (to denote wide hihghways visible from s2)
    #  - 'plant:' has power plat types (solar, etc.)
    #  -'traffic_signals' ??
    #  - 'ammenity' ??
    #  - surface ??
    #  - tunnel ??
    #  - iata / icao ??
    #  - golf ??
    #  - building:* ??
    #  - office ??
    #  - service ??
    #  - public_transport ??
    newtags = {str(k).lower():str(v) for k,v in tags.items() if v is not None}
    newtags = {k:v for k,v in newtags.items() if not k in ignore_tags and sum([k.startswith(ti) for ti in ignore_tags if ti[-1]==':'])==0}
    
    return newtags

class OSMSimpleHandler(osmium.SimpleHandler):
    
    def __init__(self, pbf_filepath, use_progress_bar=True):

        osmium.SimpleHandler.__init__(self)
        self.pbar = progressbar.ProgressBar() if use_progress_bar else None
        self.pbf_filepath = pbf_filepath
        self.nodes = {}  
        self.ways = {}

    def node(self, n):
        if self.pbar is not None:
            self.pbar.increment()

        if self.only_tag is not None and not self.only_tag in n.tags:
            return
        
        data = {}
        data['tags'] = {k:v for k,v in n.tags}
        data['tags'] = clean_tags(data['tags'])
        l = n.location
        data['geometry'] = Point(l.lon, l.lat)
        self.nodes[n.id] = data
        
    def way(self, w):
        if self.pbar is not None:
            self.pbar.increment()
        if len(w.tags)==0:
            return

        if self.only_tag is not None and not self.only_tag in w.tags:
            return
        
        data = {}
        data['tags'] = {k:v for k,v in w.tags}
        data['tags'] = clean_tags(data['tags'])
        data['is_closed'] = w.is_closed()        

        locs = []
        for node in w.nodes:
            if node.ref in self.nodes.keys():
                node = self.nodes[node.ref]
                lon, lat = node['geometry'].x, node['geometry'].y
                locs.append([lon, lat]) 

        if len(locs)>0:
            lon, lat = np.r_[locs].mean(axis=0)
            data['geometry'] = Point(lon, lat)

        self.ways[w.id] = data

    def apply(self, only_tag=None, overwrite=False):
        self.only_tag = only_tag
        self.apply_file(self.pbf_filepath)


class OSMChipHandler(osmium.SimpleHandler):
    def __init__(self, chip_id, chips_path, use_progress_bar=True):
        osmium.SimpleHandler.__init__(self)
        self.pbar = progressbar.ProgressBar() if use_progress_bar else None
        self.nodes = {}  
        self.ways = {}
        self.chips_path = chips_path
        self.chip_id = chip_id
        self.pbf_filepath = f"{chips_path}/{chip_id}.pbf"
        self.geojson_filepath = f"{chips_path}/{chip_id}.geojson"
        self.parquet_filepath = f"{chips_path}/{chip_id}.parquet"
        self.shapefile_basepath = f"{chips_path}/{chip_id}"

        with open(self.geojson_filepath) as f:
            self.boundary = sh.from_geojson(f.read())

    def node(self, n):
        if self.pbar is not None:
            self.pbar.increment()
        data = {}
        data['tags'] = {k:v for k,v in n.tags}
        l = n.location
        data['geometry'] = Point(l.lon, l.lat)
        self.nodes[n.id] = data
        
    def way(self, w):
        if self.pbar is not None:
            self.pbar.increment()
        if len(w.tags)==0:
            return
        data = {}
        data['tags'] = {k:v for k,v in w.tags}
        node_ids = [n.ref for n in w.nodes]
        data['node_ids'] = node_ids
        data['is_closed'] = w.is_closed()

        zz = gpd.GeoDataFrame([self.nodes[ni] for ni in data['node_ids']], crs=epsg4326)
        zzcal = zz.to_crs(epsg_california)
        line = sh.geometry.LineString(zzcal.geometry.values)
        data['length'] = line.length

        if data['is_closed']:
            data['geometry'] = sh.geometry.Polygon([[i.x, i.y] for i in zz.geometry.values])
            data['area'] = sh.geometry.Polygon([[i.x, i.y] for i in zzcal.geometry.values]).area
        else:
            data['geometry'] = sh.geometry.LineString(zz.geometry.values)
        
        self.ways[w.id] = data

    def apply(self, overwrite=False):

        if os.path.isfile(self.parquet_filepath) and not overwrite:
            return self
            
        self.apply_file(self.pbf_filepath)
        
        # to be called after applying a file
        # consolidates everything in a single geodataframe
        dfs = []
        if len(self.nodes)>0:
            nodesdf = pd.DataFrame({k:v for k,v in self.nodes.items() if len(v['tags'])>0}).T
            nodesdf = gpd.GeoDataFrame(nodesdf, crs=epsg4326)
            
            if self.boundary is not None:
                nodesdf = nodesdf[[i.intersects(self.boundary) for i in nodesdf.geometry]]
                nodesdf["geometry"] = [i.intersection(self.boundary) for i in nodesdf.geometry]
    
            #nodes['tags'] = [tags2str(t) for t in nodes.tags]
            nodesdf['kind'] = 'node'
            nodesdf['length'] = 0
            nodesdf['area'] = 0
            self.nodesdf = nodesdf
            dfs.append(nodesdf)

        if len(self.ways)>0:
            #ways = [[i['geometry'], i['length'], i['area'] if 'area' in i.keys() else 0, tags2str(i['tags'])] for i in self.ways.values()]
            waysdf = [[i['geometry'], i['length'], i['area'] if 'area' in i.keys() else 0, i['tags']] for i in self.ways.values()]
            waysdf = gpd.GeoDataFrame(waysdf, columns=['geometry', 'length', 'area', 'tags'], crs=epsg4326)
            waysdf['kind'] = 'way'
            try:
                if self.boundary is not None:
                    waysdf = waysdf[[i.intersects(self.boundary) for i in waysdf.geometry]]
                    waysdf["geometry"] = [i.intersection(self.boundary) for i in waysdf.geometry]
        
                self.waysdf = waysdf
                dfs.append(waysdf)
            except:
                pass

        if len(dfs)>0:
            self.osmobjects = pd.concat(dfs)
    
            # cleaup tags and remove objects with no tags
            newtags = []
            for t in self.osmobjects.tags.values:
                nt = clean_tags(t)
                newtags.append(nt)
            
            self.osmobjects['tags'] = newtags
            self.osmobjects = self.osmobjects[[len(i)>0 for i in newtags]]
    
            # save objects
            self.osmobjects.to_parquet(self.parquet_filepath)    
        else:
            print ("no objects found")
        return self

    def load_osmobjects(self):
        if not 'osmobjects' in dir(self):
            self.osmobjects = gpd.read_parquet(self.parquet_filepath)        

    def save_shapefiles(self):
        zz = self.osmobjects
        zzpnt = zz[[isinstance(i, sh.geometry.point.Point) for i in zz.geometry.values]]
        zzlin = zz[[isinstance(i, sh.geometry.linestring.LineString) for i in zz.geometry.values]]
        zzpol = zz[[isinstance(i, sh.geometry.polygon.Polygon) for i in zz.geometry.values]]
        zzpnt.to_file(f"{self.shapefile_basepath}-points.shp.zip", driver='ESRI Shapefile')
        zzlin.to_file(f"{self.shapefile_basepath}-lines.shp.zip", driver='ESRI Shapefile')
        zzpol.to_file(f"{self.shapefile_basepath}-polygons.shp.zip", driver='ESRI Shapefile')

# ---------- old stuff --------------
class OSMAOI:
    
    def __init__(self, aoi, katana_threshold=1, timeout=60, key='natural'):
        self.aoi = aoi
        self.katana_threshold = katana_threshold
        self.timeout = int(timeout)
        self.key = key
        
        self.geoms = katana(self.aoi, katana_threshold)
        self.geom = sh.geometry.GeometryCollection(self.geoms)
        self.nodes = {}#get_region_hash(g):[] for g in self.geoms}
        self.ways  = {}#get_region_hash(g):[] for g in self.geoms}
        self.rels  = {}#get_region_hash(g):[] for g in self.geoms}     
        
    def getobjs(self, objtype):
        if not objtype in ['node', 'way', 'rel']:
            raise ValueError(f"invalid obj type {objtype}")
            
        area_coords = lambda x: " ".join([f"{lat:.3f} {lon:.3f}" for lon, lat in list(x.boundary.coords)])            
        
        h = get_region_hash
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

