from pathlib import Path
import shutil, os
import numpy as np
from numcodecs import Blosc
from bioio_ome_zarr.writers.channel import Channel

# Import writers + helpers
from bioio_ome_zarr.writers.ome_zarr_writer_v2 import OMEZarrWriter as OMEZarrWriterV2
from bioio_ome_zarr.writers.ome_zarr_writer_v3 import OMEZarrWriterV3
from bioio_ome_zarr.writers.utils import compute_level_shapes, compute_level_chunk_sizes_zslice

# --------------------------
# Config: TCZYX demo dataset
# --------------------------
shape = (2, 3, 1, 64, 96)     # (T, C, Z, Y, X)
dtype = np.uint8
data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

# Three levels, downsample Y/X by 2 each level
scaling = (1, 1, 1, 2, 2)
num_levels = 3

# Compute per-level shapes/chunks (legacy V2 style)
level_shapes = compute_level_shapes(shape, ["t","c","z","y","x"], scaling, num_levels)  # list[TCZYX]
level_chunks = compute_level_chunk_sizes_zslice(level_shapes)                            # list[TCZYX chunks]

# v2 compressor (numcodecs)
compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

# Output paths
p_old = Path("zarr_v2_from_old_writer.zarr")
p_new = Path("zarr_v2_from_new_writer.zarr")
for p in (p_old, p_new):
    shutil.rmtree(p, ignore_errors=True)

# -----------------------------
# 1) Write with OLD V2 writer
# -----------------------------
w_old = OMEZarrWriterV2()
w_old.init_store(
    output_path=str(p_old),
    shapes=level_shapes,
    chunk_sizes=level_chunks,
    dtype=dtype,
    compressor=compressor,
)
# write everything at once; V2 writer will build the pyramid internally
w_old.write_t_batches_array(data, channels=[], tbatch=4)

# (Old path still uses its own metadata generator/writer)
physical_scale = {"t": 1.0, "c": 1.0, "z": 1.0, "y": 1.0, "x": 1.0}
physical_units = {"x": "micrometer", "y": "micrometer", "z": "micrometer", "t": "minute"}
channel_names  = [f"C:{i}" for i in range(shape[1])]
channel_colors = [0xFFFFFF for _ in range(shape[1])]

meta_old = w_old.generate_metadata(
    image_name="TEST",
    channel_names=channel_names,
    physical_dims=physical_scale,
    physical_units=physical_units,
    channel_colors=channel_colors,
)
w_old.write_metadata(meta_old)

# -----------------------------
# 2) Write with NEW writer in v2 mode
#    (metadata comes from __init__ params)
# -----------------------------

# Convert your channel info to Channel objects (hex strings)
channels = [
    Channel(label=f"C:{i}", color=f"{color:06x}")  # e.g., "ffffff"
    for i, color in enumerate(channel_colors)
]

# Axes scales/units as ordered lists matching axes_names
axes_names = ["t","c","z","y","x"]
axes_scale = [physical_scale.get(ax, 1.0) for ax in axes_names]
axes_units = [physical_units.get(ax) for ax in axes_names]

w_new = OMEZarrWriterV3(
    store=str(p_new),
    shape=shape,
    dtype=dtype,
    scale_factors=scaling,        # per-axis factors
    axes_names=axes_names,
    axes_scale=axes_scale,        # physical pixel size at level 0 per axis
    axes_units=axes_units,        # optional units
    num_levels=num_levels,
    zarr_format=2,                # <-- write Zarr v2 (NGFF 0.4)
    image_name="TEST",
    channels=channels,            # unified Channel objects
    compressor=compressor,        # numcodecs for v2
    # Optional viewer hints (same structure works for v2/v3)
    rdefs={"defaultT": 0, "defaultZ": max(0, (shape[2] // 2) - 1), "model": "color"},
    creator_info={"tool": "bioio_ome_zarr", "version": "dev"},
)
w_new.write_full_volume(data)

# Done. No generate/write metadata calls needed; the writer wrote it during initialization.
print("Wrote:", p_old, "and", p_new)
