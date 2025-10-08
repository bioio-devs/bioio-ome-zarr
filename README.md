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
    level_shapes=(64, 64),   # (Y, X)
    dtype=data.dtype,
)

# Write the data to the store
writer.write_full_volume(data)
```

#### 5D (TCZYX), with one extra resolution level

```python
from bioio_ome_zarr.writers import OMEZarrWriter, Channel
import numpy as np

level_shapes = [
    (2, 3, 4, 256, 256),  # L0 full res
    (2, 3, 4, 128, 128),  # L1 downsampled Y/X by 2
]

data = np.random.randint(0, 255, size=level_shapes[0], dtype=np.uint8)
channels = [Channel(label=f"c{i}", color="FF0000") for i in range(data.shape[1])]

writer = OMEZarrWriter(
    store="output.zarr",
    level_shapes=level_shapes,
    dtype=data.dtype,
    zarr_format=3,  # 2 for Zarr v2
    channels=channels,
    axes_names=["t", "c", "z", "y", "x"],
    axes_types=["time", "channel", "space", "space", "space"],
    axes_units=[None, None, "micrometer", "micrometer", "micrometer"],
)

writer.write_full_volume(data)
```

### Full writer parameters and API

| Parameter               | Type                                                      | Description                                                                                                                                                 |                          |
| ----------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| **store**               | `str` or `zarr.storage.StoreLike`                         | Filesystem path, fsspec URL, or Store-like for the root group.                                                                                              |                          |
| **level_shapes**        | `Sequence[int]` **or** `Sequence[Sequence[int]]`          | Either a single N‑D shape (one level) **or** an explicit per‑level list of shapes (level 0 first). Examples above.                                          |                          |
| **dtype**               | `np.dtype` or `str`                                       | On‑disk dtype (e.g., `uint8`, `uint16`).                                                                                                                    |                          |
| **chunk_shape**         | `Sequence[int]` or `Sequence[Sequence[int]]` or `None`    | Chunk shape: single (applied to all levels) or per‑level. If `None`, a ≈16 MiB chunk is suggested per level via `multiscale_chunk_size_from_memory_target`. |                          |
| **shard_shape**         | `Sequence[int]` or `Sequence[Sequence[int]]` or `None`    | **Zarr v3 only.** Single or per‑level shard shapes. Each shard dim must be an integer multiple of the corresponding chunk dim.                              |                          |
| **compressor**          | `BloscCodec` (v3) or `numcodecs.abc.Codec` (v2) or `None` | Compression codec. Defaults to zstd + bitshuffle.                                                                                                           |                          |
| **zarr_format**         | `Literal[2,3]`                                            | Target Zarr format – `2` (NGFF 0.4) or `3` (NGFF 0.5). Default `3`.                                                                                         |                          |
| **image_name**          | `str` or `None`                                           | Name used in multiscales metadata. Default: `"Image"`.                                                                                                      |                          |
| **channels**            | `list[Channel]` or `None`                                 | OMERO‑style channel metadata.                                                                                                                               |                          |
| **rdefs**               | `dict` or `None`                                          | OMERO rendering defaults.                                                                                                                                   |                          |
| **creator_info**        | `dict` or `None`                                          | Optional creator block (NGFF 0.5).                                                                                                                          |                          |
| **root_transform**      | `dict[str, Any]` or `None`                                | Optional transform placed at multiscale root.                                                                                                               |                          |
| **axes_names**          | `list[str]` or `None`                                     | Axis names; defaults to the last N of `["t","c","z","y","x"]`.                                                                                              |                          |
| **axes_types**          | `list[str]` or `None`                                     | Axis types; defaults to `["time","channel","space", …]`.                                                                                                    |                          |
| **axes_units**          | `list[str]`or`None`                                       | Physical units per axis. |
| **physical_pixel_size** | `list[float]` or `None`                                   | Level‑0 physical scale per axis.                                                                                                                            |                          |

**Methods**

* `write_full_volume(input_data: np.ndarray | dask.array.Array) -> None`
  Write level‑0 and all declared pyramid levels. NumPy arrays are wrapped into Dask using level‑0 chunks.

* `write_timepoints(data: np.ndarray | dask.array.Array, *, start_T_src=0, start_T_dest=0, total_T: int | None = None) -> None`
  Stream along the **T** axis from `data` into the store. Spatial axes are downsampled for lower levels; **T**/**C** are preserved.

* `preview_metadata() -> dict[str, Any]`
  Returns the NGFF metadata dict(s) that would be written (no IO).

### Writing a full volume (NumPy or Dask)

```python
# NumPy (wrapped automatically)
writer.write_full_volume(data)

# Or pass an explicit Dask array
import dask.array as da
writer.write_full_volume(da.from_array(data, chunks=(1, 1, 1, 64, 64)))
```

### Writing timepoints in batches (streaming along T)

```python
# Suppose your writer axes include "T"; write timepoints in flexible batches
from bioio import BioImage
import dask.array as da

bioimg = BioImage("/path/to/any/bioimage")
data = bioimg.get_image_dask_data()

# Write the entire timeseries at once
writer.write_timepoints(data)

# Write in 5-timepoint batches
for t in range(0, data.shape[0], 5):
    writer.write_timepoints(
    data,
    start_T_src=t,
    start_T_dest=t,
    total_T=5,
    )

# Write source timepoints [10:20] into destination positions [50:60]
writer.write_timepoints(
    data,
    start_T_src=10,
    start_T_dest=50,
    total_T=10,
)
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
    level_shapes=[(1, 1, 2, 256, 256), (1, 1, 2, 128, 128)],
    dtype="uint16",
    zarr_format=3,
    chunk_shape=chunk_shape,
)

# Example data matching the declared shape
arr = np.random.randint(0, 65535, size=(1, 1, 2, 256, 256), dtype=np.uint16)
writer.write_full_volume(arr)
```

### Sharded arrays (v3 only)

```python
from zarr.codecs import BloscCodec, BloscShuffle

writer = OMEZarrWriter(
    store="sharded_v3.zarr",
    level_shapes=[(1, 1, 16, 1024, 1024), (1, 1, 16, 512, 512)],
    dtype="uint8",
    zarr_format=3,
    chunk_shape=[(1, 1, 1, 128, 128),(1, 1, 1, 128, 128)],
    shard_shape=[(1, 1, 1, 256, 256), (1, 1, 1, 256, 256)],
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
    level_shapes=[(2, 1, 4, 256, 256), (2, 1, 4, 128, 128)],
    dtype="uint8",
    zarr_format=2,  # write NGFF 0.4
    compressor=numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
    ),
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
    level_shapes=(1, 2, 8, 2048, 2048),  # single level (TCZYX), no pyramid
    dtype="uint16",
    zarr_format=3,
)

writer.write_full_volume(
    np.random.randint(0, 65535, size=(1, 2, 8, 2048, 2048), dtype=np.uint16)
)
```

---

### Writer Utility Functions

**`multiscale_chunk_size_from_memory_target(level_shapes, dtype, memory_target) -> list[tuple[int, ...]]`**  
Suggests **per-level** chunk shapes that each fit within a fixed byte budget.

- Works for any ndim (2…5).
- **prioritizes the highest-index axis first** (grow X, then Y, then Z, then C, then T).

### Example: 16 MiB budget on large pyramids (rightmost-axis first)

```python
from bioio_ome_zarr.writers.utils import multiscale_chunk_size_from_memory_target

# 4D (C, Z, Y, X) across 5 levels
level_shapes = [
    (8, 64, 4096, 4096),
    (8, 64, 2048, 2048),
    (8, 64, 1024, 1024),
    (8, 64,  512,  512),
    (8, 64,  256,  256),
]

# 16 MiB target
chunks = multiscale_chunk_size_from_memory_target(level_shapes, "uint16", 16 << 20)

chunks = [
   (1,  1, 2048, 4096),
   (1,  2, 2048, 2048),
   (1,  8, 1024, 1024),
   (1, 32,  512,  512),
   (2, 64,  256,  256),
 ]
```

**`add_zarr_level(existing_zarr, scale_factors, compressor=None, t_batch=4) -> None`**
Appends a new resolution level to an existing **v2 OME-Zarr** store, writing in time (`T`) batches.

* `scale_factors`: per-axis scale relative to the previous highest level (tuple of length 5 for `T, C, Z, Y, X`).
* Automatically determines appropriate chunk size using `multiscale_chunk_size_from_memory_target`.
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
### Using Config Presets

Config presets make it easy to get started with `OMEZarrWriter` without needing to know all of its options. They inspect your input data and return a configuration dictionary that you can pass directly into the writer.

#### Visualization preset

The visualization preset (`get_default_config_for_viz`) creates a multiscale pyramid (full resolution plus downsampled levels along Y/X) suitable for interactive browsing.

```python
import numpy as np
from bioio_ome_zarr.writers import (
    OMEZarrWriter,
    get_default_config_for_viz,
)

data = np.zeros((1, 1, 4, 64, 64), dtype="uint16")

cfg = get_default_config_for_viz(data)
writer = OMEZarrWriter("output.zarr", **cfg)
writer.write_full_volume(data)
```

This produces a Zarr store with the original data and additional lower-resolution levels for visualization.

#### Machine learning preset

The ML preset (`get_default_config_for_ml`) writes only the full-resolution data, chunked to optimize for patch-wise access often used in training pipelines.


## 🚨 Deprecation Notice

The legacy **OmeZarrWriterV2** class (referred to here as “V2 Writer”) and **OmeZarrWriterV3** class (referred to here as “V3 Writer”) are **deprecated** and will be removed in a future release.
They has been replaced by the new **OMEZarrWriter**, which supports writing to **both Zarr v2 (NGFF 0.4)** and **Zarr v3 (NGFF 0.5)** formats.

For new code, please **use OMEZarrWriter**.

---


## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch) or check this repository's issue tab.


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

