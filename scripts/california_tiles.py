from pathlib import Path

import boto3
import geopandas as gpd
import numpy
import pandas as pd
import rasterio
from rasterio.windows import Window, bounds, transform
from shapely import box

wd = Path("~/Desktop/california").expanduser()

# Download worldcover layers over california
grid = gpd.read_file(
    "s3://clay-california-worldcover-rgbnir-vvvh-chips/esa_wordlcover_grid_california.fgb"
)
s3 = boto3.client("s3")

for id, tile in grid.iterrows():
    print(
        "esa-worldcover-s2",
        tile.s2_rgbnir_2021.split("s3://esa-worldcover-s2/")[1],
        f"{wd}/rgbnir/{tile.s2_rgbnir_2021.split('/')[-1]}",
    )
    s3.download_file(
        "esa-worldcover-s2",
        tile.s2_rgbnir_2021.split("s3://esa-worldcover-s2/")[1],
        f"{wd}/rgbnir/{tile.s2_rgbnir_2021.split('/')[-1]}",
    )


for id, tile in grid.iterrows():
    print(
        "esa-worldcover-s1",
        tile.s1_vvvhratio_2021.split("s3://esa-worldcover-s1/")[1],
        f"{wd}/vvvhratio/{tile.s1_vvvhratio_2021.split('/')[-1]}",
    )
    s3.download_file(
        "esa-worldcover-s1",
        tile.s1_vvvhratio_2021.split("s3://esa-worldcover-s1/")[1],
        f"{wd}/vvvhratio/{tile.s1_vvvhratio_2021.split('/')[-1]}",
    )

# Merge all into very large files
"""
gdal_merge.py -co BIGTIFF=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 -co TILED=YES -co COMPRESS=DEFLATE -o ${wd}/california_rgbnir_worldcover.tif ${wd}/rgbnir/*.tif
gdal_merge.py -co BIGTIFF=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 -co TILED=YES -co COMPRESS=DEFLATE -o ${wd}/california_vvvhratio_worldcover.tif ${wd}/vvvhratio/*.tif
"""

# Create chips from merged files
RGB = wd / "california_rgbnir_worldcover.tif"
VVVH = wd / "california_vvvhratio_worldcover.tif"
TILE_SIZE = 256
NO_DATA = 0
count = 0
boxes = []
cols = []
rows = []
with rasterio.open(RGB) as rgbnir:
    meta = rgbnir.meta.copy()
    meta["count"] = 6
    meta["width"] = TILE_SIZE
    meta["height"] = TILE_SIZE
    meta["compress"] = "deflate"
    with rasterio.open(VVVH) as vvvh:
        for i in range(0, rgbnir.width, TILE_SIZE):
            for j in range(0, rgbnir.height, TILE_SIZE):
                dst_path = wd / f"chips/worldcover_california_chip_{i}_{j}.tif"
                # if dst_path.exists():
                #     continue
                win = Window(i, j, TILE_SIZE, TILE_SIZE)
                meta["transform"] = transform(win, rgbnir.transform)
                rgbnir_chip = rgbnir.read([1, 2, 3, 4], window=win)
                if 0 in rgbnir_chip.shape:
                    continue
                if rgbnir_chip.shape[1] != TILE_SIZE:
                    continue
                if rgbnir_chip.shape[2] != TILE_SIZE:
                    continue
                # Filter at 10% nodata
                if numpy.sum(rgbnir_chip[2] == NO_DATA) > (TILE_SIZE * TILE_SIZE) / 10:
                    continue
                vvvh_chip = vvvh.read([1, 2], window=win)
                count += 1
                data = numpy.vstack([rgbnir_chip, vvvh_chip])
                with rasterio.open(dst_path, "w", **meta) as dst:
                    dst.write(data)
                # Track bounding boxes
                boxes.append(box(*bounds(win, rgbnir.transform)))
                cols.append(i)
                rows.append(j)
            print(f"Done with column {i} / {rgbnir.width}")


chips = gpd.GeoDataFrame(
    pd.DataFrame(
        {
            "col": cols,
            "row": rows,
        }
    ),
    crs="EPSG:4326",
    geometry=boxes,
)

chips.to_file(
    "s3://clay-california-worldcover-rgbnir-vvvh-chips/california-worldcover-chips.fgb"
)


# Upload chips to s3
"""
s5cmd sync ${wd}/chips/ s3://clay-california-worldcover-rgbnir-vvvh-chips/chips/
"""
