import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

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
    if not cts:
        cts = [{}]
    cts[0]["type"] = cts[0].get("type", "scale")
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
