import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numcodecs
import numpy as np
import skimage.transform
import zarr
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from zarr.storage import LocalStore

from bioio_ome_zarr.reader import Reader

from .metadata import OME_NGFF_VERSION_V04, OME_NGFF_VERSION_V05, Channel

DimSeq = Sequence[int]
PerLevelDimSeq = Sequence[DimSeq]


@dataclass
class ZarrLevel:
    """Descriptor for a Zarr multiscale level."""

    shape: DimSeq
    chunk_size: DimSeq
    dtype: np.dtype
    zarray: zarr.Array


def resize(
    image: da.Array, output_shape: Tuple[int, ...], *args: Any, **kwargs: Any
) -> da.Array:
    factors = np.array(output_shape) / np.array(image.shape, float)
    better_chunksize = tuple(
        np.maximum(1, np.ceil(np.array(image.chunksize) * factors) / factors).astype(
            int
        )
    )
    image_prepared = image.rechunk(better_chunksize)

    block_output_shape = tuple(
        np.ceil(np.array(better_chunksize) * factors).astype(int)
    )

    def resize_block(image_block: da.Array, block_info: dict) -> da.Array:
        chunk_output_shape = tuple(
            np.ceil(np.array(image_block.shape) * factors).astype(int)
        )
        return skimage.transform.resize(
            image_block, chunk_output_shape, *args, **kwargs
        ).astype(image_block.dtype)

    output_slices = tuple(slice(0, d) for d in output_shape)
    output = da.map_blocks(
        resize_block, image_prepared, dtype=image.dtype, chunks=block_output_shape
    )[output_slices]
    return output.rechunk(image.chunksize).astype(image.dtype)


def add_zarr_level(
    existing_zarr: Union[str, Path],
    scale_factors: Tuple[float, float, float, float, float],  # (T, C, Z, Y, X)
    compressor: Optional[numcodecs.abc.Codec] = None,
    t_batch: int = 4,
) -> None:
    """
    Append one more resolution level to an OME-Zarr, writing in T-slices.
    """
    rdr = Reader(existing_zarr)
    levels = list(rdr.resolution_levels)
    if not levels:
        raise RuntimeError("No existing resolution levels found.")

    src_idx = max(levels)
    src_shape = rdr.resolution_level_dims[src_idx]
    dtype = rdr.dtype

    new_shape = tuple(int(np.ceil(s * f)) for s, f in zip(src_shape, scale_factors))
    chunks = multiscale_chunk_size_from_memory_target(
        [new_shape], dtype, 16 * 1024 * 1024
    )[0]
    store = LocalStore(str(existing_zarr))
    group = zarr.open_group(store=store, mode="a", zarr_format=2)
    new_idx = src_idx + 1
    arr = group.create_array(
        name=str(new_idx),
        shape=new_shape,
        chunks=chunks,
        dtype=dtype,
        compressors=[compressor] if compressor is not None else None,
        overwrite=False,
        fill_value=0,
        chunk_key_encoding=V2ChunkKeyEncoding(separator="/").to_dict(),
    )

    total_t = src_shape[0]
    for t_start in range(0, total_t, t_batch):
        t_end = min(t_start + t_batch, total_t)
        t_block = rdr.get_image_dask_data(
            "TCZYX", resolution_level=src_idx, T=slice(t_start, t_end)
        )
        resized = resize(t_block, (t_end - t_start, *new_shape[1:]), order=0).astype(
            dtype
        )

        da.to_zarr(
            resized,
            arr,
            region=(slice(t_start, t_end),) + (slice(None),) * (resized.ndim - 1),
            overwrite=True,
        )

    ms = group.attrs.get("multiscales", [{}])
    datasets = ms[0].setdefault("datasets", [])
    if datasets:
        last_scale = datasets[-1]["coordinateTransformations"][0]["scale"]
    else:
        last_scale = [1] * len(scale_factors)
    new_scale = [p * f for p, f in zip(last_scale, scale_factors)]

    datasets.append(
        {
            "path": str(new_idx),
            "coordinateTransformations": [
                {"type": "scale", "scale": new_scale},
                {"type": "translation", "translation": [0] * len(scale_factors)},
            ],
        }
    )
    group.attrs["multiscales"] = ms

    print(
        f"Added level {new_idx}: shape={new_shape}, chunks={chunks}, scale={new_scale}"
    )


def pyramid_levels_to_tile_target(
    level0_shape: Tuple[int, ...],
    canvas_size: int = 2048,
    n_spatial: int = 2,
) -> List[Tuple[int, ...]]:
    """
    Build pyramid level shapes by halving until all Z planes, arranged in a
    grid, fit within a ``canvas_size × canvas_size`` pixel canvas.

    Each Z plane is one tile of size Y × X. Tiles need not be square. The
    bottom level is the largest (least-downsampled) level where Z tiles can
    be arranged in *some* rows × cols grid that satisfies
    ``rows * Y <= canvas_size`` and ``cols * X <= canvas_size``. This is
    equivalent to ``(canvas_size // Y) * (canvas_size // X) >= Z``.

    Dtype is not considered — this is a pixel-dimension constraint only.
    Level 0 is always included verbatim.

    Y and X are halved at every step. Z is also halved whenever it exceeds
    the *largest* of the new Y and X (i.e. ``Z > max(Y//2, X//2)``), which
    prevents the small dimension of non-square tiles from triggering premature
    Z halving.
    With ``n_spatial=2`` there is no Z axis, the grid is always 1×1, and the
    condition reduces to ``Y <= canvas_size`` and ``X <= canvas_size``.

    Examples (n_spatial=3, canvas_size=2048)::

        # Z=1 — 1×1 grid; constraint is simply Y <= 2048 and X <= 2048
        (1, 4096, 4096) -> (1, 2048, 2048)

        # Z=4 — square tiles, 2×2 grid fits at 1024
        (4, 2048, 2048) -> (4, 1024, 1024)

        # Z=9 — constraint: (2048//Y)*(2048//X) >= 9; halving stops at 512
        (9, 2048, 2048) -> (9, 1024, 1024) -> (9, 512, 512)

        # Z=50 — square tiles; stops at 256
        (50, 8192, 8192) -> ... -> (50, 256, 256)

        # Non-square: Z=4, wide tiles (Y=512, X=2048) — 4×1 layout fits at level 0
        # (2048//512)*(2048//2048) = 4*1 = 4 >= 4  → already fits
        (4, 512, 2048) -> [(4, 512, 2048)]

        # Non-square: Z=50, tall tiles — optimal layout drives the stopping level
        (50, 4096, 512) -> (50, 2048, 256) -> (50, 1024, 128) -> (50, 512, 64)

        # Level 0 already fits — returned as-is
        (1, 512, 512) -> [(1, 512, 512)]

    Parameters
    ----------
    level0_shape:
        Shape of the top (full-resolution) level.
    canvas_size:
        Total pixel budget for the Z-plane grid in each spatial dimension.
        Default 2048.
    n_spatial:
        Number of rightmost axes treated as spatial. Default 2 (Y/X only,
        single-tile grid). Set to 3 for ZYX data to enable Z-grid tiling.

    Returns
    -------
    List[Tuple[int, ...]]
        Pyramid level shapes from level 0 down to the first level whose Z-plane
        grid fits within ``canvas_size``. Returns a single-element list if
        level 0 already fits.
    """
    ndim = len(level0_shape)
    spatial_indices = list(range(ndim - min(n_spatial, ndim), ndim))
    inner_indices = (
        spatial_indices[-2:] if len(spatial_indices) > 2 else spatial_indices
    )
    outer_indices = spatial_indices[:-2] if len(spatial_indices) > 2 else []

    def _next_shape(current: Tuple[int, ...]) -> Tuple[int, ...]:
        shape = list(current)
        min_outer = min(current[i] for i in outer_indices) if outer_indices else 0
        for i in inner_indices:
            shape[i] = max(1, shape[i] // 2)
        # Halve Z only when it exceeds the largest remaining spatial dimension.
        # Using max(Y, X) rather than min(Y, X) prevents extreme-aspect-ratio tiles
        # from triggering premature Z halving via the small dimension.
        next_max_inner = max((shape[i] for i in inner_indices), default=1)
        if min_outer > next_max_inner:
            for i in outer_indices:
                shape[i] = max(1, shape[i] // 2)
        return tuple(shape)

    def _fits(shape: Tuple[int, ...]) -> bool:
        z = math.prod(shape[i] for i in outer_indices) if outer_indices else 1
        y = shape[inner_indices[0]] if len(inner_indices) >= 2 else 1
        x = shape[inner_indices[-1]] if inner_indices else 1
        # Maximum rows and cols that fit one tile within the canvas; check that
        # their product can accommodate all z planes in some arrangement.
        return (canvas_size // y) * (canvas_size // x) >= z

    levels: List[Tuple[int, ...]] = [tuple(int(x) for x in level0_shape)]
    if _fits(levels[0]):
        return levels

    while True:
        prev = levels[-1]
        next_shape = _next_shape(prev)
        if next_shape == prev:
            break
        levels.append(next_shape)
        if _fits(next_shape):
            break

    return levels


def multiscale_chunk_size_from_memory_target(
    level_shapes: Sequence[Sequence[int]],
    dtype: Union[str, np.dtype],
    memory_target: int,
) -> List[Sequence[int]]:
    """
    Compute per-level chunk shapes under a fixed byte budget, **prioritizing the
    highest-index axis first** (i.e., grow X, then Y, then Z, ... moving left).

    Note
    -----
    These chunk sizes represent an **in-memory target only** and do not account
    for compression.

    Returns
    -------
    list[Sequence[int]]
        Per-level chunk shapes (same ndim/order as the input).
    """
    if not level_shapes:
        raise ValueError("level_shapes cannot be empty")

    # Validation
    for i, shp in enumerate(level_shapes):
        if len(shp) < 2:
            raise ValueError(f"level_shapes[{i}] must have ndim >= 2, got {len(shp)}")
        if any(int(d) < 1 for d in shp):
            raise ValueError(f"level_shapes[{i}] has non-positive dimension(s): {shp}")

    itemsize = np.dtype(dtype).itemsize
    if memory_target < itemsize:
        raise ValueError(f"memory_target {memory_target} < dtype size {itemsize}")

    budget_elems = max(1, memory_target // itemsize)

    out: List[Sequence[int]] = []
    for shp in level_shapes:
        # Start with all-ones, grow from rightmost axis leftward
        chunk = [1] * len(shp)
        for axis in reversed(range(len(shp))):
            used = math.prod(chunk)
            if used >= budget_elems:
                # No room left; keep remaining (left) axes at 1
                break
            max_here = budget_elems // used
            # Cap by the level dimension and ensure at least 1
            chunk[axis] = max(1, min(int(shp[axis]), int(max_here)))
        out.append(tuple(chunk))

    return out


def _get_ds_scale(ds: Dict[str, Any]) -> Optional[List[float]]:
    cts = ds.get("coordinateTransformations") or []
    if not cts:
        return None
    s = cts[0].get("scale")
    if s is None:
        return None
    return [float(x) for x in s]


def _set_ds_scale(ds: Dict[str, Any], scale: List[float]) -> None:
    cts = ds.get("coordinateTransformations") or [{}]
    cts[0].setdefault("type", "scale")
    cts[0]["scale"] = list(scale)
    ds["coordinateTransformations"] = cts


def _propagate_physical_pixel_size(
    ms0: Dict[str, Any], base_scale: List[float]
) -> None:
    """
    Rewrite datasets[*].coordinateTransformations[0].scale such that:
      new_scale_l = base_scale * (old_scale_l / old_scale_0)
    """
    datasets = list(ms0.get("datasets") or [])
    if not datasets:
        return

    old0 = _get_ds_scale(datasets[0])
    if old0 is None or len(old0) != len(base_scale):
        for ds in datasets:
            _set_ds_scale(ds, base_scale)
        ms0["datasets"] = datasets
        return

    for ds in datasets:
        oldl = _get_ds_scale(ds)
        if oldl is None or len(oldl) != len(base_scale):
            _set_ds_scale(ds, base_scale)
            continue

        ratio: List[float] = [
            (ol / o0) if o0 != 0 else 1.0 for ol, o0 in zip(oldl, old0)
        ]
        newl = [bs * r for bs, r in zip(base_scale, ratio)]
        _set_ds_scale(ds, newl)

    ms0["datasets"] = datasets


def _infer_axes_len_from_ms0(ms0: Dict[str, Any]) -> int:
    """
    Infer how many axes we should have.

    Priority:
      1) datasets[0].coordinateTransformations[0].scale length
      2) existing ms0["axes"] length
      3) 0
    """
    datasets = ms0.get("datasets") or []
    if datasets:
        s = _get_ds_scale(datasets[0])
        if s is not None:
            return len(s)

    axes = ms0.get("axes") or []
    if axes:
        return len(axes)

    return 0


def _apply_axes_edits(
    ms0: Dict[str, Any],
    *,
    axes_names: Optional[List[str]],
    axes_types: Optional[List[str]],
    axes_units: Optional[List[Optional[str]]],
) -> None:
    """
    Apply axis edits by index. If the store has no axes yet, create placeholders
    long enough to accept the provided edits (or inferred ndim).

    Intentionally brittle: no validation / consistency checks.
    """
    if axes_names is None and axes_types is None and axes_units is None:
        return

    axes = list(ms0.get("axes") or [])

    # If no axes exist yet, create placeholders long enough
    if not axes:
        want = max(
            len(axes_names or []),
            len(axes_types or []),
            len(axes_units or []),
            _infer_axes_len_from_ms0(ms0),
        )
        if want <= 0:
            return
        axes = [{"name": f"dim_{i}"} for i in range(want)]

    # Apply edits by index
    for i in range(len(axes)):
        if axes_names is not None and i < len(axes_names):
            axes[i]["name"] = axes_names[i]
        if axes_types is not None and i < len(axes_types):
            axes[i]["type"] = axes_types[i]
        if axes_units is not None and i < len(axes_units):
            u = axes_units[i]
            if u is None:
                axes[i].pop("unit", None)
            else:
                axes[i]["unit"] = u

    ms0["axes"] = axes


# ----------------------------
# V3 handler (NGFF 0.5)
# ----------------------------


def _edit_metadata_v3(
    root: zarr.Group,
    *,
    image_name: Optional[str] = None,
    channels: Optional[List[Channel]] = None,
    rdefs: Optional[dict] = None,
    creator_info: Optional[dict] = None,
    root_transform: Optional[Dict[str, Any]] = None,
    axes_names: Optional[List[str]] = None,
    axes_types: Optional[List[str]] = None,
    axes_units: Optional[List[Optional[str]]] = None,
    physical_pixel_size: Optional[List[float]] = None,
) -> None:
    """
    Apply edits to a Zarr v3 / NGFF 0.5 store in-place.
    Expects `root` opened in mode="a".
    """
    ome = dict(root.attrs.get("ome", {}))
    ome.setdefault("version", OME_NGFF_VERSION_V05)

    ome.setdefault("multiscales", [{}])
    if not ome["multiscales"]:
        ome["multiscales"] = [{}]

    ms0 = dict(ome["multiscales"][0])

    if image_name is not None:
        ms0["name"] = image_name

    if root_transform is not None:
        ms0["coordinateTransformations"] = [root_transform]

    _apply_axes_edits(
        ms0,
        axes_names=axes_names,
        axes_types=axes_types,
        axes_units=axes_units,
    )

    if physical_pixel_size is not None:
        _propagate_physical_pixel_size(ms0, list(physical_pixel_size))

    ome["multiscales"][0] = ms0

    # OMERO (optional)
    if channels is not None or rdefs is not None:
        omero = dict(ome.get("omero", {}))
        omero.setdefault("version", OME_NGFF_VERSION_V05)
        if channels is not None:
            omero["channels"] = [ch.to_dict() for ch in channels]
        if rdefs is not None:
            omero["rdefs"] = rdefs
        ome["omero"] = omero

    # Creator (optional)
    if creator_info is not None:
        ome["_creator"] = creator_info

    root.attrs["ome"] = ome


# ----------------------------
# V2 handler (NGFF 0.4)
# ----------------------------


def _edit_metadata_v2(
    root: zarr.Group,
    *,
    image_name: Optional[str] = None,
    channels: Optional[List[Channel]] = None,
    rdefs: Optional[dict] = None,
    root_transform: Optional[Dict[str, Any]] = None,
    axes_names: Optional[List[str]] = None,
    axes_types: Optional[List[str]] = None,
    axes_units: Optional[List[Optional[str]]] = None,
    physical_pixel_size: Optional[List[float]] = None,
) -> None:
    """
    Apply edits to a Zarr v2 / NGFF 0.4 store in-place.
    Expects `root` opened in mode="a".
    """
    multiscales = list(root.attrs.get("multiscales") or [])
    if not multiscales:
        multiscales = [{}]

    ms0 = dict(multiscales[0])

    if image_name is not None:
        ms0["name"] = image_name

    if root_transform is not None:
        ms0["coordinateTransformations"] = [root_transform]

    _apply_axes_edits(
        ms0,
        axes_names=axes_names,
        axes_types=axes_types,
        axes_units=axes_units,
    )

    if physical_pixel_size is not None:
        _propagate_physical_pixel_size(ms0, list(physical_pixel_size))

    ms0.setdefault("version", OME_NGFF_VERSION_V04)
    multiscales[0] = ms0
    root.attrs["multiscales"] = multiscales

    # OMERO (v2 root-level)
    if channels is not None or rdefs is not None or image_name is not None:
        omero = dict(root.attrs.get("omero") or {})
        omero.setdefault("version", OME_NGFF_VERSION_V04)
        if image_name is not None:
            omero["name"] = image_name
        if channels is not None:
            omero["channels"] = [ch.to_dict() for ch in channels]
        if rdefs is not None:
            omero["rdefs"] = rdefs
        root.attrs["omero"] = omero


# ----------------------------
# Dispatcher
# ----------------------------


def edit_metadata(
    store: Union[str, Path, zarr.storage.StoreLike],
    *,
    image_name: Optional[str] = None,
    channels: Optional[List[Channel]] = None,
    rdefs: Optional[dict] = None,
    creator_info: Optional[dict] = None,
    root_transform: Optional[Dict[str, Any]] = None,
    axes_names: Optional[List[str]] = None,
    axes_types: Optional[List[str]] = None,
    axes_units: Optional[List[Optional[str]]] = None,
    physical_pixel_size: Optional[List[float]] = None,
) -> None:
    """
    Apply metadata edits to an existing OME-Zarr store in-place.

    This function opens the provided Zarr store and updates
    metadata fields. The appropriate metadata layout is selected
    automatically based on the presence of NGFF v3-style
    ``root.attrs["ome"]`` metadata.

    Parameters
    ----------
    store: Union[str, Path, zarr.storage.StoreLike]
        Path to the Zarr store or an existing Zarr store object.
        The store must already contain OME-NGFF multiscale metadata.
    image_name: Optional[str]
        New image name to assign to the root multiscale entry.
    channels: Optional[List[Channel]]
        Channel metadata to write into the OMERO metadata block.
        Each ``Channel`` is serialized using ``Channel.to_dict()``.
    rdefs: Optional[dict]
        Rendering definitions to write into the OMERO metadata block.
    creator_info: Optional[dict]
        Creator or provenance metadata (NGFF v3 only). Stored under
        the ``ome`` metadata block.
    root_transform: Optional[Dict[str, Any]]
        Root-level coordinate transformation dictionary to assign to
        the multiscale entry (e.g., a scale or translation transform).
    axes_names: Optional[List[str]]
        List of axis names to assign by index.
    axes_types: Optional[List[str]]
        List of axis types to assign by index (e.g., "space", "time", "channel").
    axes_units: Optional[List[Optional[str]]]
        List of axis units to assign by index. ``None`` removes the unit
        field for that axis.
    physical_pixel_size: Optional[List[float]]
        Base physical pixel size to apply to the multiscale pyramid.
        Existing per-level scale values are updated proportionally to
        preserve relative downsampling factors.

    Returns
    -------
    None
        Metadata is updated in-place within the provided store.
    """
    root = zarr.open_group(store, mode="a")

    if isinstance(root.attrs.get("ome"), dict):
        _edit_metadata_v3(
            root,
            image_name=image_name,
            channels=channels,
            rdefs=rdefs,
            creator_info=creator_info,
            root_transform=root_transform,
            axes_names=axes_names,
            axes_types=axes_types,
            axes_units=axes_units,
            physical_pixel_size=physical_pixel_size,
        )
    else:
        _edit_metadata_v2(
            root,
            image_name=image_name,
            channels=channels,
            rdefs=rdefs,
            root_transform=root_transform,
            axes_names=axes_names,
            axes_types=axes_types,
            axes_units=axes_units,
            physical_pixel_size=physical_pixel_size,
        )
