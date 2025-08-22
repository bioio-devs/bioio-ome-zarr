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

ChunkSingle = Tuple[int, ...]
ChunkPerLevel = Tuple[ChunkSingle, ...]
ChunkShape = Union[ChunkSingle, ChunkPerLevel]


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
        chunk_shape: Optional[ChunkShape] = None,
        shard_factor: Optional[Tuple[int, ...]] = None,
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
        shard_factor : Optional[Tuple[int, ...]]
            Optional shard factor per axis (v3 only); ignored for v2.
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
            # Detect single-tuple vs tuple-of-tuples
            if len(chunk_shape) > 0 and isinstance(chunk_shape[0], int):
                single = cast(ChunkSingle, chunk_shape)
                if len(single) != self.ndim:
                    raise ValueError(
                        f"chunk_shape length {len(single)} != ndim {self.ndim}"
                    )
                self.chunk_shapes_per_level = [
                    tuple(int(x) for x in single) for _ in range(self.num_levels)
                ]
            else:
                per_level = cast(ChunkPerLevel, chunk_shape)
                if len(per_level) != self.num_levels:
                    raise ValueError(
                        f"chunk_shape must match levels: ({self.num_levels}); "
                        f"got {len(per_level)}"
                    )
                shapes: List[Tuple[int, ...]] = []
                for idx, ch in enumerate(per_level):
                    if len(ch) != self.ndim:
                        raise ValueError(
                            f"chunk_shape[{idx}] length {len(ch)} != ndim {self.ndim}"
                        )
                    shapes.append(tuple(int(x) for x in ch))
                self.chunk_shapes_per_level = shapes
        else:
            suggested = chunk_size_from_memory_target(
                self.level_shapes[0], self.dtype, 16 << 20
            )
            self.chunk_shapes_per_level = [suggested for _ in range(self.num_levels)]

        # 5) formatting and compression
        self.zarr_format = zarr_format
        self.shard_factor = shard_factor
        self.compressor = compressor

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
        source: Union[np.ndarray, da.Array],
        *,
        tbatch: int = 1,
    ) -> None:
        """
        Batch write along T. Source must already match the writer’s axis
        order and dimensionality. Spatial axes are downsampled for lower
        levels; T and C are preserved.
        """
        if not self._initialized:
            self._initialize()

        writer_axes = [a.lower() for a in self.axes.names]
        if "t" not in writer_axes:
            raise ValueError(
                "write_timepoints() requires a time axis 'T' in writer axes."
            )

        axis_t = writer_axes.index("t")
        total_T = int(self.level_shapes[0][axis_t])
        tbatch = max(1, int(tbatch))

        # Wrap numpy in dask if needed
        if isinstance(source, np.ndarray):
            chunks = self.datasets[0].chunks
            assert chunks is not None, "Writer datasets must have chunks."
            arr = da.from_array(source, chunks=chunks)
        else:
            arr = source  # already dask

        if arr.ndim != self.ndim:
            raise ValueError(
                f"Array source ndim ({arr.ndim}) must match writer.ndim ({self.ndim})."
            )

        for start_t in range(0, total_T, tbatch):
            end_t = min(start_t + tbatch, total_T)
            if end_t <= start_t:
                continue

            # Slice this time batch once from the input
            sel_batch: List[slice] = [slice(None)] * self.ndim
            sel_batch[axis_t] = slice(start_t, end_t)
            block = arr[tuple(sel_batch)]

            # Process each level independently to avoid accumulating resized arrays
            for lvl in range(self.num_levels):
                if lvl == 0:
                    level_block = block
                else:
                    nextshape = list(self.level_shapes[lvl])
                    nextshape[axis_t] = end_t - start_t
                    level_block = resize(block, tuple(nextshape), order=0).astype(
                        block.dtype
                    )

                for k_abs in range(start_t, end_t):
                    rel = k_abs - start_t
                    sel_one: List[slice] = [slice(None)] * self.ndim
                    sel_one[axis_t] = slice(rel, rel + 1)
                    subset = level_block[tuple(sel_one)]

                    region = tuple(
                        slice(k_abs, k_abs + 1) if i == axis_t else slice(None)
                        for i in range(self.ndim)
                    )
                    if self.zarr_format == 2:
                        da.to_zarr(subset, self.datasets[lvl], region=region)
                    else:
                        # Zarr v3
                        da.store(
                            subset,
                            self.datasets[lvl],
                            regions=region,
                            lock=True,
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
                if self.shard_factor is not None:
                    kwargs["shards"] = tuple(
                        c * f for c, f in zip(chunks_lvl, self.shard_factor)
                    )
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
