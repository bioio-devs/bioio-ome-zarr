from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import dask.array as da
import numcodecs
import numpy as np
import zarr
from numcodecs import Blosc as BloscV2
from zarr.codecs import BloscCodec, BloscShuffle

from .metadata import Axes, Channel, MetadataParams, build_ngff_metadata
from .utils import (
    chunk_size_from_memory_target,
    compute_level_chunk_sizes_zslice,
    resize,
)

DimTuple = Tuple[int, ...]
PerLevelDimTuple = List[DimTuple]
LevelwiseSpec = Union[DimTuple, PerLevelDimTuple]


def _normalize_levelwise(
    spec: LevelwiseSpec,
    *,
    num_levels: int,
    ndim: int,
    label: str,
) -> List[Tuple[int, ...]]:
    """
    Convert a single N-dim tuple or a per-level tuple-of-tuples into a
    per-level List[Tuple[int, ...]] with validation.
    """
    if len(spec) == 0:
        raise ValueError(f"{label} cannot be empty")
    # single-tuple case
    if isinstance(spec[0], int):  # type: ignore[index]
        single = cast(DimTuple, spec)  # type: ignore[assignment]
        if len(single) != ndim:
            raise ValueError(f"{label} length {len(single)} != ndim {ndim}")
        return [tuple(int(x) for x in single) for _ in range(num_levels)]
    # per-level case
    per_level = cast(PerLevelDimTuple, spec)
    if len(per_level) != num_levels:
        raise ValueError(
            f"{label} must have {num_levels} entries (per level), got {len(per_level)}"
        )
    out: List[Tuple[int, ...]] = []
    for i, t in enumerate(per_level):
        if len(t) != ndim:
            raise ValueError(f"{label}[{i}] length {len(t)} != ndim {ndim}")
        out.append(tuple(int(v) for v in t))
    return out


class OMEZarrWriter:
    """
    OMEZarrWriter is a unified OME-Zarr writer that targets either Zarr v2
    (NGFF 0.4) or Zarr v3 (NGFF 0.5) with the same public API. Supports
    2 ≤ N ≤ 5 dimensions (e.g., YX, ZYX, TYX, CZYX, or TCZYX) and writes a
    multiscale pyramid with nearest-neighbor downsampling.
    """

    def __init__(
        self,
        store: Union[str, zarr.storage.StoreLike],
        shape: Tuple[int, ...],
        dtype: Union[np.dtype, str],
        *,
        scale: Optional[Tuple[Tuple[float, ...], ...]] = None,
        chunk_shape: Optional[LevelwiseSpec] = None,
        shard_shape: Optional[LevelwiseSpec] = None,
        compressor: Optional[Union[BloscCodec, numcodecs.abc.Codec]] = None,
        zarr_format: Literal[2, 3] = 3,
        image_name: Optional[str] = "Image",
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
        Initialize the writer and capture core configuration. Arrays and
        metadata are created lazily on the first write. Does not write to
        disk until data is written.

        Parameters
        ----------
        store : Union[str, zarr.storage.StoreLike]
            Filesystem path, URL (via fsspec), or Store-like for the root
            group.
        shape : Tuple[int, ...]
            Level-0 image shape (e.g., (T,C,Z,Y,X)).
        dtype : Union[np.dtype, str]
            NumPy dtype for the on-disk array.
        scale : Optional[Tuple[Tuple[float, ...], ...]]
            Per-level, per-axis *relative size* vs. level-0. For example,
            ``((1,1,0.5,0.5,0.5), (1,1,0.25,0.25,0.25))`` writes two extra
            levels at 1/2 and 1/4 resolution on spatial axes. If ``None``,
            only level-0 is written.
        chunk_shape : Optional[Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]]
            Either a single chunk shape (applied to all levels),
            e.g. ``(1,1,16,256,256)``, or per-level chunk shapes,
            e.g. ``((1,1,16,256,256), (1,1,16,128,128), ...)``. If ``None``,
            a suggested ≈16 MiB chunk is derived from level-0 and reused;
            for Zarr v2, if omitted, a legacy per-level policy may be applied
            when axes are TCZYX.
        shard_shape : Optional[LevelwiseSpec]
            **Zarr v3 only.** Either:
              - a single N-dim tuple applied to all levels, or
              - a per-level tuple-of-tuples.
            Ignored for Zarr v2.
        compressor : Optional[BloscCodec | numcodecs.abc.Codec]
            Compression codec. For v2 use ``numcodecs.Blosc``; for v3 use
            ``zarr.codecs.BloscCodec``.
        zarr_format : Literal[2,3]
            Target Zarr array format: 2 (NGFF 0.4) or 3 (NGFF 0.5).
        image_name : Optional[str]
            Image name used in multiscales metadata. Default: "Image".
        channels : Optional[List[Channel]]
            OMERO-style channel metadata objects.
        rdefs : Optional[dict]
            Optional OMERO rendering defaults.
        creator_info : Optional[dict]
            Optional creator block placed in metadata (v0.5).
        root_transform : Optional[Dict[str, Any]]
            Optional multiscale root coordinate transformation.
        axes_names : Optional[List[str]]
            Axis names; defaults to last N of ["t","c","z","y","x"].
        axes_types : Optional[List[str]]
            Axis types; defaults to ["time","channel","space", …].
        axes_units : Optional[List[Optional[str]]]
            Physical units for each axis.
        physical_pixel_size : Optional[List[float]]
            Physical scale at level 0 for each axis.
        """
        # 1) Store fundamental properties
        self.store = store
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)

        # 2) Build an Axes instance
        self.axes = Axes(
            ndim=self.ndim,
            names=axes_names,
            types=axes_types,
            units=axes_units,
            scales=physical_pixel_size,
            factors=tuple(1 for _ in range(self.ndim)),
        )

        # 3) Compute all pyramid level shapes from `scale`
        self.level_shapes: List[Tuple[int, ...]] = [tuple(self.shape)]
        self.dataset_scales: List[List[float]] = []

        if scale is not None:
            for level_scale in scale:
                if len(level_scale) != self.ndim:
                    raise ValueError(
                        f"Each scale tuple must have length {self.ndim}; "
                        f"got {len(level_scale)}"
                    )

            # Normalize to the declared type (List[List[float]])
            self.dataset_scales = [list(map(float, tpl)) for tpl in scale]
            for vec in self.dataset_scales:
                next_shape = tuple(
                    max(1, int(np.floor(self.shape[i] * vec[i])))
                    for i in range(self.ndim)
                )
                if next_shape == self.level_shapes[-1]:
                    continue
                self.level_shapes.append(next_shape)

        self.num_levels = len(self.level_shapes)

        # 4) Determine per-level chunk shapes
        self._chunk_shape_explicit = chunk_shape is not None
        if chunk_shape is not None:
            self.chunk_shapes_per_level = _normalize_levelwise(
                chunk_shape,
                num_levels=self.num_levels,
                ndim=self.ndim,
                label="chunk_shape",
            )
        else:
            suggested = chunk_size_from_memory_target(
                self.level_shapes[0], self.dtype, 16 << 20
            )
            self.chunk_shapes_per_level = [suggested for _ in range(self.num_levels)]

        # 5) formatting and compression
        self.zarr_format = zarr_format
        self.compressor = compressor
        self.shards_per_level: Optional[List[Tuple[int, ...]]] = None
        if shard_shape is not None:
            self.shards_per_level = _normalize_levelwise(
                shard_shape,
                num_levels=self.num_levels,
                ndim=self.ndim,
                label="shard_shape",
            )
        else:
            self.shards_per_level = None

        # 6) Metadata fields
        self.image_name = image_name or "Image"
        self.channels = channels
        self.rdefs = rdefs
        self.creator_info = creator_info
        self.root_transform = root_transform

        # 8) Handles & state
        self.root: Optional[zarr.Group]
        self.datasets: List[zarr.Array]
        self._initialized: bool = False
        self._metadata_written: bool = False

    # -----------------
    # Public interface
    # -----------------
    def preview_metadata(self) -> Dict[str, Any]:
        """
        Build and return the exact NGFF metadata dict(s) this writer will
        persist. Safe to call before initializing the store; uses in-memory
        config/state.
        """
        params = MetadataParams(
            image_name=self.image_name,
            axes=self.axes,
            level_shapes=self.level_shapes,
            channels=self.channels,
            rdefs=self.rdefs,
            creator_info=self.creator_info,
            root_transform=self.root_transform,
            dataset_scales=self.dataset_scales,
        )
        return build_ngff_metadata(
            zarr_format=self.zarr_format,
            params=params,
        )

    def write_full_volume(
        self,
        input_data: Union[np.ndarray, da.Array],
    ) -> None:
        """
        Write full-resolution data into all pyramid levels.

        Parameters
        ----------
        input_data : Union[np.ndarray, dask.array.Array]
            Array matching level-0 shape. If NumPy, it will be wrapped into a
            Dask array with level-0 chunking.
        """
        if not self._initialized:
            self._initialize()

        base = (
            input_data
            if isinstance(input_data, da.Array)
            else da.from_array(input_data, chunks=self.datasets[0].chunks)
        )

        # Store each level (downsampled with nearest-neighbor for parity)
        for lvl, shape in enumerate(self.level_shapes):
            src = base if lvl == 0 else resize(base, shape, order=0)
            if self.zarr_format == 2:
                da.to_zarr(src, self.datasets[lvl])
            else:
                da.store(src, self.datasets[lvl], lock=True)

    def write_timepoints(
        self,
        data: Union[np.ndarray, da.Array],
        *,
        start_T_src: int = 0,
        start_T_dest: int = 0,
        total_T: Optional[int] = None,
    ) -> None:
        """
        Write a contiguous batch of timepoints from `data` into all pyramid levels.

        Parameters
        ----------
        data : np.ndarray | dask.array.Array
            Array in writer axis order containing the source timepoints.
            If a NumPy array is provided, it is minimally wrapped as a Dask
            array with ``chunks="auto"``. For optimal performance and IO
            alignment, pass a Dask array with explicit chunks.
        start_T_src : int, optional
            Source T index at which to begin reading from `data`. Default: 0.
        start_T_dest : int, optional
            Destination T index at which to begin writing into the store. Default: 0.
        total_T : int, optional
            Number of timepoints to transfer. If None, inferred as the maximum
            that fits within both the source (from ``start_T_src``) and destination
            (from ``start_T_dest``).
        """
        if not self._initialized:
            self._initialize()

        # Ensure valid dims
        writer_axes = [a.lower() for a in self.axes.names]
        if "t" not in writer_axes:
            raise ValueError("write_t_batch() requires a 'T' axis.")
        axis_t = writer_axes.index("t")
        arr = (
            da.from_array(data, chunks="auto") if isinstance(data, np.ndarray) else data
        )
        if arr.ndim != self.ndim:
            raise ValueError(
                f"Array ndim ({arr.ndim}) must match writer.ndim ({self.ndim})."
            )

        src_T = int(arr.shape[axis_t])
        dst_T = int(self.level_shapes[0][axis_t])
        if start_T_src >= src_T or start_T_dest >= dst_T:
            return

        src_avail = src_T - start_T_src
        dst_avail = dst_T - start_T_dest

        # Infer or bound total_T
        if total_T is None:
            total_T = min(src_avail, dst_avail)
        else:
            total_T = max(0, min(int(total_T), src_avail, dst_avail))
        if total_T == 0:
            return

        # Source slice
        sel_src: List[slice] = [slice(None)] * self.ndim
        sel_src[axis_t] = slice(start_T_src, start_T_src + total_T)
        batch_arr = arr[tuple(sel_src)]

        # Destination region slice
        region_tuple = tuple(
            slice(start_T_dest, start_T_dest + total_T) if i == axis_t else slice(None)
            for i in range(self.ndim)
        )

        # Per-level processing
        for lvl in range(self.num_levels):
            if lvl == 0:
                level_block = batch_arr
            else:
                nextshape = list(self.level_shapes[lvl])
                nextshape[axis_t] = total_T
                level_block = resize(batch_arr, tuple(nextshape), order=0).astype(
                    batch_arr.dtype
                )

            # Dispatch write depending on Zarr format
            if self.zarr_format == 2:
                da.to_zarr(level_block, self.datasets[lvl], region=region_tuple)
            else:
                da.store(
                    level_block, self.datasets[lvl], regions=region_tuple, lock=True
                )

    # -----------------
    # Internal plumbing
    # -----------------

    def _initialize(self) -> None:
        """
        Open the root group, create arrays for each level, and write metadata
        once. Subsequent writes reuse the created arrays.
        """
        self.root = self._open_root()

        if self.compressor is None:
            if self.zarr_format == 2:
                compressor = BloscV2(
                    cname="zstd",
                    clevel=3,
                    shuffle=BloscV2.BITSHUFFLE,
                )
            else:
                compressor = BloscCodec(
                    cname="zstd",
                    clevel=3,
                    shuffle=BloscShuffle.bitshuffle,
                )
        else:
            compressor = self.compressor

        self.datasets = []

        if self.zarr_format == 2:
            # v2
            if not self._chunk_shape_explicit:
                # If 5D TCZYX, use legacy z-slice per-level chunking; otherwise
                # keep the suggested per-level chunking already prepared.
                is_tczyx = self.ndim == 5 and [n.lower() for n in self.axes.names] == [
                    "t",
                    "c",
                    "z",
                    "y",
                    "x",
                ]
                if is_tczyx:
                    self.chunk_shapes_per_level = compute_level_chunk_sizes_zslice(
                        self.level_shapes
                    )

            for lvl, shape in enumerate(self.level_shapes):
                chunks_lvl = self.chunk_shapes_per_level[lvl]
                arr = self.root.zeros(
                    name=str(lvl),
                    shape=shape,
                    chunks=chunks_lvl,
                    dtype=self.dtype,
                    compressor=compressor,
                    zarr_format=2,
                    dimension_separator="/",
                )
                self.datasets.append(arr)
        else:
            # v3
            for lvl, shape in enumerate(self.level_shapes):
                chunks_lvl = self.chunk_shapes_per_level[lvl]
                kwargs: Dict[str, Any] = {
                    "name": str(lvl),
                    "shape": shape,
                    "chunks": chunks_lvl,
                    "dtype": self.dtype,
                    "compressors": compressor,
                    "chunk_key_encoding": {
                        "name": "default",
                        "separator": "/",
                    },
                }
                # Per-level shards if provided (Zarr v3 only)
                if self.shards_per_level is not None:
                    kwargs["shards"] = tuple(int(x) for x in self.shards_per_level[lvl])

                arr = self.root.create_array(**kwargs)
                self.datasets.append(arr)

        # Write metadata
        self._write_metadata()
        self._metadata_written = True

        self._initialized = True

    def _open_root(self) -> zarr.Group:
        """Accept a path/URL or Store-like and return an opened root group."""
        if isinstance(self.store, str):
            if "://" in self.store:
                fs = zarr.storage.FsspecStore(self.store, mode="w")
                return zarr.open_group(store=fs, mode="w", zarr_format=self.zarr_format)
            return zarr.open_group(self.store, mode="w", zarr_format=self.zarr_format)
        return zarr.group(
            store=self.store,
            overwrite=True,
            zarr_format=self.zarr_format,
        )

    def _write_metadata(self) -> None:
        """Persist NGFF metadata to the opened root group."""
        if self.root is None:
            raise RuntimeError("Store must be initialized before writing metadata.")

        md = self.preview_metadata()
        if self.zarr_format == 2:
            self.root.attrs["multiscales"] = md["multiscales"]
            self.root.attrs["omero"] = md["omero"]
        else:
            self.root.attrs.update({"ome": md["ome"]})
