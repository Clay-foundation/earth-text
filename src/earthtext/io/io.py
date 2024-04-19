import xarray as xr
import geopandas as gpd
import os
import torch
import pickle

def read_multilabel_metadata(filename):
    if filename.endswith('.parquet'):
        return gpd.read_parquet(filename)

    raise ValueError("unknown file format for multilabel metadata")

def check_chip_exists(chips_folder, embeddings_folder, patch_embeddings_folder, col, row):

    # if no chips or embeddings are needed we do not need to check the files
    if chips_folder is None and embeddings_folder is None and patch_embeddings_folder is None:
        return True

    # otherwise we require the chips to exist if the corresponding folder is define

    r = True

    chip_fname             = f"{chips_folder}/worldcover_california_chip_{col}_{row}.tif"
    embeddings_fname       = f"{embeddings_folder}/worldcover_california_chip_{col}_{row}.pkl"
    patch_embeddings_fname = f"{patch_embeddings_folder}/worldcover_california_chip_{col}_{row}.pkl"

    r = r and (chips_folder is None or os.path.isfile(chip_fname))
    r = r and (embeddings_folder is None or os.path.isfile(embeddings_fname))
    r = r and (patch_embeddings_folder is None or os.path.isfile(patch_embeddings_fname))

    return r

def read_chip(chips_folder, col, row):
    fname = f"{chips_folder}/worldcover_california_chip_{col}_{row}.tif"
    
    with xr.open_dataarray(fname) as z:
        chip = z.data.copy()

    return torch.tensor(chip).type(torch.int32)

def read_embedding(embeddings_folder, col, row):
    fname = f"{embeddings_folder}/worldcover_california_chip_{col}_{row}.pkl"
    with open(fname, "rb") as f:
        r = pickle.load(f)
    return r

def read_patch_embedding(patch_embeddings_folder, col, row):
    fname = f"{patch_embeddings_folder}/worldcover_california_chip_{col}_{row}.pkl"
    with open(fname, "rb") as f:
        r = pickle.load(f)
    return r