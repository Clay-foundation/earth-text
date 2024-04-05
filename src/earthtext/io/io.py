import xarray as xr
import geopandas as gpd
import os
import torch

def read_multilabel_metadata(filename):
    if filename.endswith('.parquet'):
        return gpd.read_parquet(filename)

    raise ValueError("unknown file format for multilabel metadata")

def check_chip_exists(chips_folder, col, row):
    fname = f"{chips_folder}/worldcover_california_chip_{col}_{row}.tif"

    return os.path.isfile(fname)

def read_chip(chips_folder, col, row):
    fname = f"{chips_folder}/worldcover_california_chip_{col}_{row}.tif"
    
    with xr.open_dataarray(fname) as z:
        chip = z.data.copy()

    return torch.tensor(chip).type(torch.int32)
