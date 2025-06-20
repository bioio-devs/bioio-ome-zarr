import os
import shutil

import numpy as np
import zarr
from bioio_ome_zarr.writers import Channel, OmeZarrWriterV3

# NOTE: This script will load image completely into memory and is not meant for large files 

# Replace with your actual paths:
input_store = "/Users/brian.whitney/Desktop/Repos/bioio-ome-zarr/bioio_ome_zarr/tests/resources/dimension_handling_tyx.zarr"
output_store = "/Users/brian.whitney/Desktop/Repos/bioio-ome-zarr/bioio_ome_zarr/tests/resources/dimension_handling_tyx_V3.zarr"

# If the output directory already exists, remove it so we start fresh
if os.path.exists(output_store):
    shutil.rmtree(output_store)

# 1) Open the input OME-Zarr v2 store
v2_group = zarr.open_group(input_store, mode="r")

# 2) Read the root attributes (V2 places “multiscales” and “omero” at top level)
v2_attrs = v2_group.attrs.asdict()

# 3) Extract the first (and only) multiscale entry, including its “name”
ms_v2 = v2_attrs["multiscales"][0]
image_name_v2 = ms_v2.get("name", "")

# 3a) Extract top-level scale if present (a list of floats)
multiscale_scale = None
for tf in ms_v2.get("coordinateTransformations", []):
    if tf.get("type") == "scale" and "scale" in tf:
        multiscale_scale = tf["scale"]
        break

# 4) Gather axes info from V2
axes_v2 = ms_v2["axes"]
axes_names = [ax["name"] for ax in axes_v2]
axes_types = [ax["type"] for ax in axes_v2]
axes_units = [ax.get("unit", None) for ax in axes_v2]
ndim = len(axes_names)

# 5) Read the “scale” vector at level 0
datasets_v2 = ms_v2["datasets"]
scale0 = datasets_v2[0]["coordinateTransformations"][0]["scale"]

# 6) Determine how many levels exist in V2
declared_num_levels = len(datasets_v2)

# 7) Identify truly distinct levels (dropping any whose shape does not change):
valid_v2_levels = [0]
lvl0_shape_raw = v2_group["0"].shape  # e.g. (Z,Y,X)
if len(lvl0_shape_raw) < ndim:
    pad = (1,) * (ndim - len(lvl0_shape_raw))
    prev_shape = pad + lvl0_shape_raw
else:
    prev_shape = lvl0_shape_raw

for level in range(1, declared_num_levels):
    shape_raw = v2_group[str(level)].shape
    if len(shape_raw) < ndim:
        pad = (1,) * (ndim - len(shape_raw))
        shape_n = pad + shape_raw
    else:
        shape_n = shape_raw

    if shape_n != prev_shape:
        valid_v2_levels.append(level)
        prev_shape = shape_n
    else:
        break

num_levels = len(valid_v2_levels)

# 8) Recompute “scale1” from the first two kept levels; otherwise reuse scale0
if num_levels > 1:
    lvl1_idx = valid_v2_levels[1]
    scale1 = datasets_v2[lvl1_idx]["coordinateTransformations"][0]["scale"]
else:
    scale1 = scale0[:]

# 9) Build scale_factors by dividing scale1[i]/scale0[i] for “space” axes
scale_factors_list = []
for i in range(ndim):
    if axes_types[i] == "space":
        factor = int(scale1[i] / scale0[i])
        scale_factors_list.append(max(1, factor))
    else:
        scale_factors_list.append(1)
scale_factors = tuple(scale_factors_list)

# 10) For V3, axes_scale = exactly V2’s level-0 “scale” vector
axes_scale = scale0[:]

# 11) Read level 0 Zarr array to get its raw shape and dtype
lvl0_arr = v2_group["0"]
lvl0_shape_raw = lvl0_arr.shape
dtype = lvl0_arr.dtype

# 12) Pad the level 0 shape so that full_shape has length = ndim
if len(lvl0_shape_raw) < ndim:
    pad = (1,) * (ndim - len(lvl0_shape_raw))
    full_shape = pad + lvl0_shape_raw
else:
    full_shape = lvl0_shape_raw

# 13) Extract chunk sizes from level 0 (if present), then pad chunks to length = ndim
chunks0 = lvl0_arr.chunks  # e.g. (chunk_z, chunk_y, chunk_x)
if chunks0 is not None and len(chunks0) < ndim:
    pad_chunks = (1,) * (ndim - len(chunks0))
    chunks = pad_chunks + chunks0
else:
    chunks = chunks0

# 14) Leave shards=None so V3 picks defaults
shards = None

# 15) Extract OMERO‐style channel metadata (if present)
channels_meta = v2_attrs.get("omero", {}).get("channels", [])
channels_list = [Channel(**ch_meta) for ch_meta in channels_meta] \
    if channels_meta else None

# 16) Determine whether V2’s axes include "c"; if so, pad or tile accordingly
full_data_raw = lvl0_arr[...]  # raw on-disk array, e.g. (Z,Y,X) or (C,Z,Y,X)
if "c" in axes_names and channels_list:
    c_idx = axes_names.index("c")
    if full_data_raw.ndim == ndim - 1:
        # raw lacks channel axis → pad, expand, tile
        pad_shape = (1,) * (ndim - full_data_raw.ndim) + full_data_raw.shape
        tmp = full_data_raw.reshape(pad_shape)
        tmp = np.expand_dims(tmp, axis=c_idx)
        reps = [1] * tmp.ndim
        reps[c_idx] = len(channels_list)
        full_data = np.tile(tmp, reps)
        # Update full_shape to include channel
        fs = list(full_shape)
        fs.insert(c_idx, len(channels_list))
        full_shape = tuple(fs)
        # Pad chunks if present
        if chunks is not None:
            cl = list(chunks)
            insert_chunk = min(len(channels_list),
                               cl[c_idx]) if c_idx < len(cl) else len(channels_list)
            cl.insert(c_idx, insert_chunk)
            chunks = tuple(cl)
    else:
        # raw already has channel axis; pad on left if needed
        if full_data_raw.ndim < ndim:
            pad = (1,) * (ndim - full_data_raw.ndim)
            full_data = full_data_raw.reshape(pad + full_data_raw.shape)
        else:
            full_data = full_data_raw
else:
    # No "c" in axes_names → treat as single-channel
    if full_data_raw.ndim < ndim:
        pad = (1,) * (ndim - full_data_raw.ndim)
        full_data = full_data_raw.reshape(pad + full_data_raw.shape)
    else:
        full_data = full_data_raw

# 17) Pass along V2’s _creator block (if present)
creator_info = v2_attrs.get("_creator", None)

# 18) Instantiate the V3 writer with exactly num_levels levels,
#     including top-level scale if found
writer = OmeZarrWriterV3(
    store=output_store,
    shape=full_shape,
    dtype=dtype,
    axes_names=axes_names,
    axes_types=axes_types,
    axes_units=axes_units,
    axes_scale=axes_scale,
    scale_factors=scale_factors,
    num_levels=num_levels,
    chunks=chunks,
    shards=shards,
    channels=channels_list,       # attach channel metadata if any
    creator_info=creator_info,
    image_name=image_name_v2,
    multiscale_scale=multiscale_scale,  # pass top-level scale list directly
)

# 19) Write out the full‐volume data → V3 will produce exactly num_levels levels
writer.write_full_volume(full_data)

print(
    f"Converted V2 '{input_store}' → V3 '{output_store}', "
    f"image_name='{image_name_v2}', {num_levels} levels, "
    f"channels_included={bool(channels_list)}, "
    f"global_scale={multiscale_scale}."
)
