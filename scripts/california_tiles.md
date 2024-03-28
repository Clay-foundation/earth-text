# California chips
This file documents how the Worldcover chips were created for the earth-text
model.

This script creates 111'920 chips with a pizel size of `256x256` pixels. Each
Chip has 5 bands: `[red, green, blue, nir, vv, vh]`, all stored as 16bit
unsigned integer.

The steps for creating the tiles were as follows:

1. Create file with worldcover grid cells over california
2. Download the worldcover data for those cells for rgbnir and vvvh
3. Merge the cells into single bigtiff files
4. Use a python script to create chips with the rgb, nir, and vvvh bands in one file
5. Create index file with bounding boxes for all these chips


## Normalizations statistics for the Sentinel-1 bands

The `vv` and  `vh` bands are in uint16, which is different from the data that the
Clay model has been trained with. The normalization parameters have to be adapted,
The statistics of the data are as follows.

### vv band statistics

```
Band 1 Block=256x256 Type=UInt16, ColorInterp=Gray
  Min=0.000 Max=56073.000 
  Minimum=4659.000, Maximum=58757.000, Mean=34369.390, StdDev=3572.343
  NoData Value=0
  Metadata:
    STATISTICS_MAXIMUM=58757
    STATISTICS_MEAN=34369.390015449
    STATISTICS_MINIMUM=4659
    STATISTICS_STDDEV=3572.3427052005
    STATISTICS_VALID_PERCENT=46.72
```

### vh band statistics

```
Band 2 Block=256x256 Type=UInt16, ColorInterp=Undefined
  Min=0.000 Max=47888.000 
  Minimum=4948.000, Maximum=58353.000, Mean=26265.214, StdDev=5056.598
  NoData Value=0
  Metadata:
    STATISTICS_MAXIMUM=58353
    STATISTICS_MEAN=26265.213704928
    STATISTICS_MINIMUM=4948
    STATISTICS_STDDEV=5056.5981159613
    STATISTICS_VALID_PERCENT=46.72
```

## Data location

### Worldcover grid cells layer

```
s3://clay-california-worldcover-rgbnir-vvvh-chips/esa_wordlcover_grid_california.fgb
```

### Index file for all the chips

```
s3://clay-california-worldcover-rgbnir-vvvh-chips/california-worldcover-chips.fgb
```

### Chips

Chips are stored under the `chips` common prefix on S3

```
s3://clay-california-worldcover-rgbnir-vvvh-chips/chips/
```
