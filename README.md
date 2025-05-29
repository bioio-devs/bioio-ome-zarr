# bioio-ome-zarr

[![Build Status](https://github.com/bioio-devs/bioio-ome-zarr/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-ome-zarr/actions)
[![PyPI version](https://badge.fury.io/py/bioio-ome-zarr.svg)](https://badge.fury.io/py/bioio-ome-zarr)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO reader plugin for reading OME ZARR images using `ome-zarr`

---


## Documentation

[See the full documentation on our GitHub pages site](https://bioio-devs.github.io/bioio/OVERVIEW.html) - the generic use and installation instructions there will work for this package.

Information about the base reader this package relies on can be found in the `bioio-base` repository [here](https://github.com/bioio-devs/bioio-base)

## Installation

**Stable Release:** `pip install bioio-ome-zarr`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-ome-zarr.git`

## Example Usage (see full documentation for more examples)

Install bioio-ome-zarr alongside bioio:

`pip install bioio bioio-ome-zarr`


This example shows a simple use case for just accessing the pixel data of the image
by explicitly passing this `Reader` into the `BioImage`. Passing the `Reader` into
the `BioImage` instance is optional as `bioio` will automatically detect installed
plug-ins and auto-select the most recently installed plug-in that supports the file
passed in.
```python
from bioio import BioImage
import bioio_ome_zarr

img = BioImage("my_file.zarr", reader=bioio_ome_zarr.Reader)
img.data
```

### Reading from AWS S3
To read from private S3 buckets, [credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) must be configured. Public buckets can be accessed without credentials.
```python
from bioio import BioImage
path = "https://allencell.s3.amazonaws.com/aics/nuc-morph-dataset/hipsc_fov_nuclei_timelapse_dataset/hipsc_fov_nuclei_timelapse_data_used_for_analysis/baseline_colonies_fov_timelapse_dataset/20200323_09_small/raw.ome.zarr"
image = BioImage(path)
print(image.get_image_dask_data())
```
## OME Zarr V2 Writer

Import the writer and utilities:

```python
from bioio_ome_zarr.writers import (
    OmeZarrWriterV2,
    chunk_size_from_memory_target,
    compute_level_shapes,
    compute_level_chunk_sizes_zslice,
    resize,
)

```

Use utility functions directly:

```python
# Compute chunk size within a memory target
shape = (1, 1, 1, 128, 128)  # TCZYX
dtype = np.uint16
mem_target = 1024
chunk = chunk_size_from_memory_target(shape, dtype, mem_target)
# chunk == (1, 1, 1, 16, 16)
```

```python
# Compute multiscale shapes
base_shape = (1, 1, 1, 128, 128)
scaling = (1.0, 1.0, 1.0, 2.0, 2.0)
levels = 2
shapes = compute_level_shapes(base_shape, scaling, levels)
# shapes == [(1,1,1,128,128), (1,1,1,64,64)]
```

```python
# Compute chunk sizes per level (Z-slice strategy)
shapes = [
    (512, 4, 100, 1000, 1000),
    (512, 4, 100,  500,  500),
    (512, 4, 100,  250,  250),
]
chunks = compute_level_chunk_sizes_zslice(shapes)
# chunks == [(1,1,1,1000,1000), (1,1,4,500,500), (1,1,16,250,250)]
```

### Writing OME-Zarr Stores

```python
# Prepare data and pyramid parameters
shape = (4, 2, 2, 64, 32)       # (T,C,Z,Y,X)
# Create a dask array from random uint8 data
import numpy as np
im_np = np.random.randint(0, 256, size=shape, dtype=np.uint8)
im = da.from_array(im_np, chunks=shape)
scaling = (1.0, 1.0, 1.0, 2.0, 2.0)
levels = 3

shapes = compute_level_shapes(shape, scaling, levels)
chunks = compute_level_chunk_sizes_zslice(shapes)

# Initialize writer
writer = OMEZarrWriter()
writer.init_store(
    output_path="output.e.zarr",
    shapes=shapes,
    chunk_sizes=chunks,
    dtype=im.dtype,
)

# Write all timepoints at once
writer.write_t_batches_array(im, channels=[], tbatch=4)

# Generate and write metadata
physical_dims = {"c":1.0, "t":1.0, "z":1.0, "y":1.0, "x":1.0}
physical_units = {"x":"micrometer","y":"micrometer","z":"micrometer","t":"minute"}
channel_names = [f"c{i}" for i in range(shape[1])]
channel_colors = [0xFFFFFF for _ in range(shape[1])]
meta = writer.generate_metadata(
    image_name="TEST",
    channel_names=channel_names,
    physical_dims=physical_dims,
    physical_units=physical_units,
    channel_colors=channel_colors,
)
writer.write_metadata(meta)
```

#### Iterative Timepoint Writing

```python
# Write one timepoint at a time
writer = OMEZarrWriter()
writer.init_store("output_iter.e.zarr", shapes, chunks, im.dtype)
for t in range(shape[0]):
    frame = im[t:t+1]  # shape (1,C,Z,Y,X)
    writer.write_t_batches_array(frame, channels=[], tbatch=1, toffset=t)
# Then generate and write metadata as above
```
## OME Zarr V3 Writer

Import the writer and utilities:

```python
from bioio_ome_zarr.writers import (
    OmeZarrWriterV2,
    chunk_size_from_memory_target,
    compute_level_shapes,
    compute_level_chunk_sizes_zslice,
    resize,
)
import numpy as np
import dask.array as da
```

### Utility Examples from Tests

```python
# Downsample a 5D array (T,C,Z,Y,X)
data = np.arange(16, dtype=np.uint8).reshape((1,1,1,4,4))
factors = (1,1,1,2,2)
out = downsample_data(data, factors)
# out.shape == (1,1,1,2,2)
```

```python
# Build axes metadata
names = ["t","c","z"]
types = ["time","channel","space"]
units = [None, None, "um"]
axes = default_axes(names, types, units)
# axes == [{'name':'t','type':'time'}, {'name':'c','type':'channel'}, {'name':'z','type':'space','unit':'um'}]
```

### Pyramid Shape Computation and Chunking

```python
# Compute level shapes
shape = (1,1,1,128,128)
writer = OmeZarrWriterV3(
    store="output_v3.zarr",
    shape=shape,
    dtype=np.uint8,
    scale_factors=(1,1,1,2,2),
    num_levels=2,
)
out_shapes = writer._compute_levels(2)
# out_shapes == [(1,1,1,128,128), (1,1,1,64,64)]

# Suggest default chunks for large level
ck = writer._suggest_chunks((1,1,1,5000,5000))
# ck == (1,1,1,4096,4096)

# Shard and chunk parameters clamp per level
shape_small = (1,1,1,4,4)
writer2 = OmeZarrWriterV3(
    store="output_v3_shard.zarr",
    shape=shape_small,
    dtype=np.uint8,
    chunks=(1,1,1,2,2),
    shards=(1,1,2,2,2)
)
# writer2.chunks and writer2.shards align with level_shapes
```

### Writing Full Volume and Metadata

```python
# Write full 5D volume to a fixed output location
shape = (2,3,4,8,8)
data = np.random.randint(0,255,size=shape,dtype=np.uint8)

writer = OmeZarrWriterV3(
    store="output_full_volume.zarr",
    shape=shape,
    dtype=data.dtype,
    axes_names=["t","c","z","y","x"],
    axes_types=["time","channel","space","space","space"],
    axes_units=[None,None,None,"um","um"],
    axes_scale=[1,1,1,0.5,0.5],
    scale_factors=(1,1,2,2,2),
    num_levels=3,
    chunks=(1,1,1,4,4),
    shards=(1,1,2,2,2),
    channel_names=[f"c{i}" for i in range(shape[1])],
    channel_colors=["FF0000"]*shape[1],
    creator_info={"name":"test","version":"0.1"},
)
writer.write_full_volume(data)
```

### Writing Single Timepoints

```python
# Write one timepoint at a time to fixed output
shape = (3,2,2,4,4)
data = np.random.randint(0,255,size=shape,dtype=np.uint8)

writer = OmeZarrWriterV3(store="output_timepoints.zarr", shape=shape, dtype=data.dtype)
for t in range(shape[0]):
    frame = data[t]  # 4D (C,Z,Y,X)
    writer.write_timepoint(t, frame)
```

## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch) or check this repository's issue tab.


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
