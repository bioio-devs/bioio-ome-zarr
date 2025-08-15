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

## Writing OME-Zarr Stores

The `OMEZarrWriter` can write **both** Zarr v2 (NGFF 0.4) and Zarr v3 (NGFF 0.5) formats.

### basic writer example (2D YX)
```python
from bioio_ome_zarr.writers import OMEZarrWriter
import numpy as np

# Minimal 2D example (Y, X)
data = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

writer = OMEZarrWriter(
    store="basic.zarr",
    shape=data.shape,   # (Y, X)
    dtype=data.dtype,
)

# Write the data to the store
writer.write_full_volume(data)
```

### Full writer parameters and API

Below is a reference of the `OMEZarrWriter` parameters, For complete details, see the constructor signature.

| Parameter                 | Type                                            | Description                                                                                                                                                              |
| ------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **store**                 | `str` or `zarr.storage.StoreLike`               | Filesystem path, URL (via `fsspec`), or Store-like object for the root group.                                                                                            |
| **shape**                 | `tuple[int, ...]`                               | Shape of the highest‑resolution (level‑0) image, e.g., `(1, 1, 4, 64, 64)`.                                                                                                |
| **dtype**                 | `np.dtype` or `str`                             | NumPy dtype for the on‑disk array (e.g., `uint8`, `uint16`).                                                                                                             |
| **scale**                 | `tuple[tuple[float, ...], ...]` or `None`       | Per‑level, per‑axis *relative sizes* vs. level‑0. Example: `((1,1,0.5,0.5,0.5), (1,1,0.25,0.25,0.25))` creates two lower‑res levels. If `None`, only level‑0 is written. |
| **chunk\_shape**          | `tuple[tuple[int, ...], ...]` or `None`         | Chunk shape per level. Example: ((1, 1, 1, 64, 64), (1, 1, 1, 32, 32))If `None`, a suggested ≈16 MiB chunk is derived for v3; v2 applies a legacy per‑level policy.                                                     |
| **shard\_factor**         | `tuple[int, ...]` or `None`                     | v3‑only: multiply chunks to form shard sizes. Ignored in v2.                                                                                                             |
| **compressor**            | `BloscCodec` or `numcodecs.abc.Codec` or `None` | Compression codec. v2: `numcodecs.Blosc`; v3: `zarr.codecs.BloscCodec`.                                                                                                  |
| **zarr\_format**          | `Literal[2, 3]`                                 | Target Zarr format — `2` (NGFF 0.4) or `3` (NGFF 0.5).    Default `3`                                                                                                               |
| **image\_name**           | `str` or `None`                                 | Name used in multiscales metadata. Default: `"Image"`.                                                                                                                   |
| **channels**              | `list[Channel]` or `None`                       | OMERO‑style channel metadata.                                                                                                             |
| **rdefs**                 | `dict` or `None`                                | OMERO rdef defaults                                                                                                            |
| **creator\_info**         | `dict` or `None`                                | Optional creator block stored in metadata (v0.5).                                                                                                                        |
| **root\_transform**       | `dict[str, Any]` or `None`                      | Optional multiscale root coordinate transform.                                                                                                                           |
| **axes\_names**           | `list[str]` or `None`                           | Names of each axis. Defaults to the last N of `["t", "c", "z", "y", "x"]`.                                                                                               |
| **axes\_types**           | `list[str]` or `None`                           | Axis types. Defaults: `["time", "channel", "space", …]`.                                                                                                                 |
| **axes\_units**           | `list[str or None]` or `None`                   | Physical units for each axis (e.g., `"micrometer"`).                                                                                                                     |
| **physical\_pixel\_size** | `list[float]` or `None`                         | Physical scale at level‑0 for each axis.                                                                                                                                 |


**Methods**
- `write_full_volume(input_data: np.ndarray | dask.array.Array) -> None`  
  Write level‑0 (and all pyramid levels) from a full array. If a NumPy array is passed, it’s wrapped as Dask using level‑0 chunking.

- `write_timepoints(source: Reader | np.ndarray | dask.array.Array, *, channel_indexes: list[int] | None = None, tbatch: int = 1) -> None`  
  Stream writes along the **T** axis in batches. Writer and source axes must match by set (order handled internally). Spatial axes are downsampled for lower levels; **T** and **C** are preserved.

- `preview_metadata() -> dict[str, Any]`  
  Returns the exact NGFF metadata dict(s) that will be persisted, without writing.

---

### Creating a writer (v3 with one extra resolution level)

```python
from bioio_ome_zarr.writers import OMEZarrWriter, Channel
import numpy as np

shape = (2, 3, 4, 8, 8)  # (T, C, Z, Y, X)
data = np.random.randint(0, 255, size=shape, dtype=np.uint8)

channels = [Channel(label=f"c{i}", color="FF0000") for i in range(shape[1])]

writer = OMEZarrWriter(
    store="output.zarr",
    shape=shape,
    dtype=data.dtype,
    zarr_format=3,  # 2 for Zarr v2
    scale=((1, 1, 0.5, 0.5, 0.5),),  # add level at half Z/Y/X
    channels=channels,
    axes_names=["t", "c", "z", "y", "x"],
    axes_types=["time", "channel", "space", "space", "space"],
    axes_units=[None, None, "micrometer", "micrometer", "micrometer"],
)

# Write the entire volume
writer.write_full_volume(data)
```

### Writing a full volume (NumPy or Dask)

```python
# NumPy array is accepted and will be wrapped into Dask automatically
writer.write_full_volume(data)

# Or with a Dask array if you already have one
import dask.array as da
writer.write_full_volume(da.from_array(data, chunks=(1, 1, 1, 8, 8)))
```

### Writing timepoints in batches (streaming along T)

```python
# Suppose your writer axes include "t"; write in batches to limit memory
from bioio_ome_zarr import Reader

rdr = Reader("/path/to/large_source.ome.zarr")
# Axes in writer and reader must match by set; order is handled by the writer
writer.write_timepoints(rdr, tbatch=4, channel_indexes=[0, 2])
```

### Custom chunking per level

```python
# Provide one chunk shape per level; must match ndim
chunk_shape = (
    (1, 1, 1, 64, 64),  # level 0
    (1, 1, 1, 32, 32),  # level 1
)
writer = OMEZarrWriter(
    store="custom_chunks.zarr",
    shape=(1, 1, 2, 256, 256),
    dtype="uint16",
    zarr_format=3,
    scale=((1, 1, 0.5, 0.5, 0.5),),
    chunk_shape=chunk_shape,
)

# Example data matching the declared shape
arr = np.random.randint(0, 65535, size=(1, 1, 2, 256, 256), dtype=np.uint16)
writer.write_full_volume(arr)
```

### Sharded writes (v3 only)

```python
from zarr.codecs import BloscCodec, BloscShuffle

writer = OMEZarrWriter(
    store="sharded_v3.zarr",
    shape=(1, 1, 16, 1024, 1024),
    dtype="uint8",
    zarr_format=3,
    scale=((1, 1, 0.5, 0.5, 0.5),),
    shard_factor=(1, 1, 2, 2, 2),  # multiply chunks to form shard sizes
    compressor=BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle),
)

writer.write_full_volume(
    np.random.randint(0, 255, size=(1, 1, 16, 1024, 1024), dtype=np.uint8)
)
```

### Targeting Zarr v2 explicitly (NGFF 0.4)

```python
import numcodecs

writer = OMEZarrWriter(
    store="target_v2.zarr",
    shape=(2, 1, 4, 256, 256),
    dtype="uint8",
    zarr_format=2,  # write NGFF 0.4
    scale=((1, 1, 0.5, 0.5, 0.5), (1, 1, 0.25, 0.25, 0.25)),
    compressor=numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE),
)

writer.write_full_volume(
    np.random.randint(0, 255, size=(2, 1, 4, 256, 256), dtype=np.uint8)
)
```

### Writing to S3 (or any fsspec URL)

```python
# Requires creds for private buckets; public can be anonymous
writer = OMEZarrWriter(
    store="s3://my-bucket/path/to/out.zarr",
    shape=(1, 2, 8, 2048, 2048),
    dtype="uint16",
    zarr_format=3,
)

writer.write_full_volume(
    np.random.randint(0, 65535, size=(1, 2, 8, 2048, 2048), dtype=np.uint16)
)
```

### Add a root transform and preview metadata

```python
writer = OMEZarrWriter(
    store="with_transform.zarr",
    shape=(1, 1, 1, 128, 128),
    dtype="uint8",
    zarr_format=3,
    root_transform={"type": "scale", "scale": [1.0, 1.0, 1.0, 0.1, 0.1]},
)
# preview metadata (no disk write)
md = writer.preview_metadata()
print(md)

# Write file 
writer.write_full_volume(
    np.random.randint(0, 255, size=(1, 1, 1, 128, 128), dtype=np.uint8)
)


```
### Writer Utility Functions

**`chunk_size_from_memory_target(shape, dtype, memory_target, order=None) -> tuple[int, ...]`**
Suggests a chunk shape that fits within the specified memory target (in bytes).

* Spatial axes (`Z`, `Y`, `X`) start at full size; non-spatial axes start at `1`.
* All dimensions are halved until the total size (in bytes) fits within `memory_target`.
* Example:

```python
from bioio_ome_zarr.writers import chunk_size_from_memory_target
chunk_size_from_memory_target((1, 1, 16, 1024, 1024), "uint16", 16 * 1024 * 1024) # 16 mbs
# Returns: (1, 1, 4, 512, 512)
```

**`add_zarr_level(existing_zarr, scale_factors, compressor=None, t_batch=4) -> None`**
Appends a new resolution level to an existing **v2 OME-Zarr** store, writing in time (`T`) batches.

* `scale_factors`: per-axis scale relative to the previous highest level (tuple of length 5 for `T, C, Z, Y, X`).
* Automatically determines appropriate chunk size using `chunk_size_from_memory_target`.
* Updates the `multiscales` metadata block with the new level's path and transformations.
* Example:

```python
from bioio_ome_zarr.writers import add_zarr_level
add_zarr_level(
    "my_existing.zarr",
    scale_factors=(1, 1, 0.5, 0.5, 0.5),
    compressor=numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
)
```

## 🚨 Deprecation Notice

The legacy **OmeZarrWriterV2** class (referred to here as “V2 Writer”) is **deprecated** and will be removed in a future release.
It has been replaced by the new **OMEZarrWriter**, which supports writing to **both Zarr v2 (NGFF 0.4)** and **Zarr v3 (NGFF 0.5)** formats.

For new code, please **use OMEZarrWriter**.

---


## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch) or check this repository's issue tab.


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

