from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import dask.array as da
import numcodecs
import numpy as np
import zarr
from numcodecs import Blosc as BloscV2
from zarr.codecs import BloscCodec, BloscShuffle

from .metadata import Axes, Channel, MetadataParams, build_ngff_metadata
from .utils import (
    DimSeq,
    PerLevelDimSeq,
    multiscale_chunk_size_from_memory_target,
    resize,
)

MultiResolutionShapeSpec = Union[DimSeq, PerLevelDimSeq]


# ---------------------
# Helpers
# ---------------------
def _normalize_levelwise(
    spec: MultiResolutionShapeSpec,
    *,
    num_levels: int,
    ndim: int,
    label: str,
) -> List[Tuple[int, ...]]:
    """Normalize a single N-dim shape or per-level list of N-dim shapes into a
    per-level List[Tuple[int, ...]] with structural validation.

    Checks:
      - non-empty
      - detect single-shape vs per-level
      - per-level count equals `num_levels`
      - each shape length equals `ndim`
      - each dim >= 1

    * Note: when normalizing level_shapes, the length will not change
    only for shard_shape and chunk_shape where we bring them up to the length
    of level_shapes.
    """
    if len(spec) == 0:
        raise ValueError(f"{label} cannot be empty")

    # Single-shape case: replicate across levels
    if isinstance(spec[0], (int, np.integer)):
        single_shape = cast(DimSeq, spec)
        if len(single_shape) != ndim:
            raise ValueError(f"{label} length {len(single_shape)} != ndim {ndim}")
        single_tuple = tuple(dim for dim in single_shape)
        for axis_index, value in enumerate(single_tuple):
            if value < 1:
                raise ValueError(f"{label}[{axis_index}] must be >= 1")
        return [single_tuple] * num_levels

    # Per-level case
    per_level_shapes = cast(PerLevelDimSeq, spec)
    if len(per_level_shapes) != num_levels:
        raise ValueError(
            f"{label} must have {num_levels} entries (per level), got "
            f"{len(per_level_shapes)}"
        )
    normalized: List[Tuple[int, ...]] = []
    for level_index, per_level_shape in enumerate(per_level_shapes):
        if len(per_level_shape) != ndim:
            raise ValueError(
                f"{label}[{level_index}] length {len(per_level_shape)} != "
                f"ndim {ndim}"
            )
        level_tuple = tuple(dim for dim in per_level_shape)
        for axis_index, value in enumerate(level_tuple):
            if value < 1:
                raise ValueError(f"{label}[{level_index}][{axis_index}] must be >= 1")
        normalized.append(level_tuple)
    return normalized


def _validate_shapes(
    *,
    level_shapes: List[Tuple[int, ...]],
    chunk_shapes_per_level: List[Tuple[int, ...]],
    shards_per_level: Optional[List[Tuple[int, ...]]],
    zarr_format: Literal[2, 3],
) -> None:
    """Unified structural validation for chunks and shards.

    Rules:
      - All: per-level counts must match `level_shapes`; ndim must match.
      - Shards (v3 only): each shard dim must be a multiple of its chunk dim.
      - Shards (v2): forbidden.
    """
    level_count = len(level_shapes)
    if level_count == 0:
        raise ValueError("level_shapes cannot be empty")
    num_dims = len(level_shapes[0])

    def expect_per_level_count(
        label: str, seq: List[Tuple[int, ...]], expected: int
    ) -> None:
        if len(seq) != expected:
            raise ValueError(
                f"{label} must have {expected} entries (per level), got " f"{len(seq)}"
            )

    def expect_ndim_match(
        label: str,
        shape_tuple: Tuple[int, ...],
        level_index: int,
    ) -> None:
        if len(shape_tuple) != num_dims:
            raise ValueError(
                f"{label}[{level_index}] length {len(shape_tuple)} != "
                f"ndim {num_dims}"
            )

    # ---- chunk shapes ----
    expect_per_level_count("chunk_shape", chunk_shapes_per_level, level_count)
    for level_index, chunk_shape_level in enumerate(chunk_shapes_per_level):
        expect_ndim_match("chunk_shape", chunk_shape_level, level_index)

    # ---- shard shapes ----
    if shards_per_level is not None:
        if zarr_format == 2:
            raise ValueError("shard_shape is not supported for Zarr v2.")
        expect_per_level_count("shard_shape", shards_per_level, level_count)
        for level_index, shard_shape_level in enumerate(shards_per_level):
            expect_ndim_match("shard_shape", shard_shape_level, level_index)
            chunk_shape_level = chunk_shapes_per_level[level_index]
            for axis_index, (shard_dim, chunk_dim) in enumerate(
                zip(shard_shape_level, chunk_shape_level)
            ):
                if shard_dim % chunk_dim != 0:
                    raise ValueError(
                        f"shard_shape[{level_index}][{axis_index}] (= "
                        f"{shard_dim}) must be a multiple of chunk_dim "
                        f"{chunk_dim}"
                    )


# ---------------------
# Writer
# ---------------------


class OMEZarrWriter:
    """
    Unified OME-Zarr writer targeting Zarr v2 (NGFF 0.4) or v3 (NGFF 0.5).
    Supports 2–5D arrays (e.g., YX, ZYX, TYX, CZYX, TCZYX) and writes a
    multiscale pyramid exactly as specified by explicit per-level shapes.
    """

    def __init__(
        self,
        store: Union[str, zarr.storage.StoreLike],
        level_shapes: MultiResolutionShapeSpec,
        dtype: Union[np.dtype, str],
        *,
        chunk_shape: Optional[MultiResolutionShapeSpec] = None,
        shard_shape: Optional[MultiResolutionShapeSpec] = None,
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
            Filesystem path, URL (via fsspec), or Store-like for the root group.
        level_shapes : Sequence[int] | Sequence[Sequence[int]]
            Level-0 shape or explicit per-level shapes (level 0 first).
        dtype : Union[np.dtype, str]
            NumPy dtype for the on-disk array.
        chunk_shape : Optional[Union[Sequence[int], Sequence[Sequence[int]]]]
            Either a single chunk shape (applied to all levels),
            e.g. ``(1,1,16,256,256)``, or per-level chunk shapes,
            e.g. ``[(1,1,16,256,256), (1,1,16,128,128), ...]``. If ``None``,
            a suggested ≈16 MiB chunk is derived per level.
        shard_shape : Optional[Union[Sequence[int], Sequence[Sequence[int]]]]
            **Zarr v3 only.** Either:
              - a single N-dim sequence applied to all levels, or
              - a per-level sequence of N-dim sequences.
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
        if len(level_shapes) == 0:
            raise ValueError("level_shapes cannot be empty")

        self.store = store
        self.dtype = np.dtype(dtype)

        if isinstance(level_shapes[0], (int, np.integer)):
            inferred_ndim = len(cast(DimSeq, level_shapes))
            inferred_levels = 1
        else:
            inferred_ndim = len(cast(PerLevelDimSeq, level_shapes)[0])
            inferred_levels = len(cast(PerLevelDimSeq, level_shapes))

        # Normalize and clamp level dims to >= 1
        self.level_shapes = _normalize_levelwise(
            level_shapes,
            num_levels=inferred_levels,
            ndim=inferred_ndim,
            label="level_shapes",
        )

        self.shape = tuple(self.level_shapes[0])
        self.ndim = len(self.shape)
        self.num_levels = len(self.level_shapes)

        # Axes
        if axes_names is not None and len(axes_names) != self.ndim:
            raise ValueError(
                f"axes_names length {len(axes_names)} must match ndim {self.ndim}"
            )
        self.axes = Axes(
            ndim=self.ndim,
            names=axes_names,
            types=axes_types,
            units=axes_units,
            scales=physical_pixel_size,
            factors=tuple(1 for _ in range(self.ndim)),
        )

        # Relative scales written to NGFF metadata (vs level 0)
        self.dataset_scales: List[List[float]] = [
            (
                np.array(shape, dtype=float)
                / np.array(self.level_shapes[0], dtype=float)
            ).tolist()
            for shape in self.level_shapes[1:]
        ]

        # Chunk shapes (explicit or suggested ~16 MiB).
        self._chunk_shape_explicit: bool = chunk_shape is not None
        if chunk_shape is not None:
            self.chunk_shapes_per_level = _normalize_levelwise(
                chunk_shape,
                num_levels=self.num_levels,
                ndim=self.ndim,
                label="chunk_shape",
            )
        else:
            computed_chunk_shapes = multiscale_chunk_size_from_memory_target(
                self.level_shapes,
                self.dtype,
                16 << 20,  # ~16 MiB target
            )
            # normalize to List[Tuple[int, ...]]
            self.chunk_shapes_per_level = [
                tuple(chunk_dim for chunk_dim in single_chunk)
                for single_chunk in computed_chunk_shapes
            ]

        # Format & compressor
        self.zarr_format = zarr_format
        self.compressor = compressor

        # Shards (v3 only): normalize per-level
        self.shards_per_level: Optional[List[Tuple[int, ...]]] = None
        if shard_shape is not None:
            self.shards_per_level = _normalize_levelwise(
                shard_shape,
                num_levels=self.num_levels,
                ndim=self.ndim,
                label="shard_shape",
            )

        # Validate level/chunk/shard shapes
        _validate_shapes(
            level_shapes=self.level_shapes,
            chunk_shapes_per_level=self.chunk_shapes_per_level,
            shards_per_level=self.shards_per_level,
            zarr_format=self.zarr_format,
        )

        # Metadata fields
        self.image_name = image_name or "Image"
        self.channels = channels
        self.rdefs = rdefs
        self.creator_info = creator_info
        self.root_transform = root_transform

        # Handles & state
        self.root: Optional[zarr.Group]
        self.datasets: List[zarr.Array]
        self._initialized: bool = False
        self._metadata_written: bool = False

    # -----------------
    # Public interface
    # -----------------
    def preview_metadata(self) -> Dict[str, Any]:
        """
        Build and return NGFF metadata dict(s) this writer will
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

        cur = (
            input_data
            if isinstance(input_data, da.Array)
            else da.from_array(input_data, chunks=self.datasets[0].chunks)
        )

        expected_shape = tuple(self.level_shapes[0])
        if tuple(int(s) for s in cur.shape) != expected_shape:
            raise ValueError(
                "write_full_volume: input shape does not match level-0 shape. "
                f"Got {tuple(int(s) for s in cur.shape)} vs expected {expected_shape}."
            )

        ops = []
        for level_index, level_shape in enumerate(self.level_shapes):
            if level_index > 0:
                # Iterative downsample from previous level
                cur = resize(cur, tuple(level_shape), order=0)

            # Align dask to destination chunking
            tgt_chunks = self.datasets[level_index].chunks
            src = cur if cur.chunks == tgt_chunks else cur.rechunk(tgt_chunks)

            # lazily add
            if self.zarr_format == 2:
                ops.append(da.to_zarr(src, self.datasets[level_index], compute=False))
            else:
                ops.append(
                    da.store(src, self.datasets[level_index], lock=True, compute=False)
                )

        # compute and let dask optimize
        da.compute(*ops)

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

        writer_axes = [a.lower() for a in self.axes.names]
        if "t" not in writer_axes:
            raise ValueError("write_timepoints() requires a 'T' axis.")
        axis_t = writer_axes.index("t")

        arr = (
            da.from_array(data, chunks="auto") if isinstance(data, np.ndarray) else data
        )

        # Validate ndim & non-T dims
        if arr.ndim != self.ndim:
            raise ValueError(
                f"write_timepoints: array ndim ({arr.ndim}) "
                f"must match writer.ndim ({self.ndim})."
            )
        level0 = tuple(self.level_shapes[0])
        for ax in range(self.ndim):
            if ax == axis_t:
                continue
            got, exp = int(arr.shape[ax]), int(level0[ax])
            if got != exp:
                raise ValueError(
                    "write_timepoints: non-T axes must match destination "
                    f"level-0 shape. Axis {ax}: got {got}, expected {exp}."
                )

        src_T = int(arr.shape[axis_t])
        dst_T = int(level0[axis_t])

        # Validate starts
        if not (0 <= start_T_src < src_T):
            raise ValueError(
                f"write_timepoints: start_T_src ({start_T_src}) "
                f"out of range [0, {src_T})."
            )
        if not (0 <= start_T_dest < dst_T):
            raise ValueError(
                f"write_timepoints: start_T_dest ({start_T_dest}) "
                f"out of range [0, {dst_T})."
            )

        # Validate total_T
        src_avail = src_T - start_T_src
        dst_avail = dst_T - start_T_dest
        total_T = min(src_avail, dst_avail) if total_T is None else int(total_T)
        if total_T <= 0:
            raise ValueError("write_timepoints: total_T must be > 0.")
        if total_T > src_avail:
            raise ValueError(
                "write_timepoints: requested total_T exceeds available source "
                f"timepoints from start_T_src. Requested {total_T}, "
                f"available {src_avail}."
            )
        if total_T > dst_avail:
            raise ValueError(
                "write_timepoints: requested total_T exceeds available destination "
                f"space from start_T_dest. Requested {total_T}, available {dst_avail}."
            )

        # Source slice
        sel_src: List[slice] = [slice(None)] * self.ndim
        sel_src[axis_t] = slice(start_T_src, start_T_src + total_T)
        batch_arr = arr[tuple(sel_src)]

        # Destination region slice
        region_tuple = tuple(
            slice(start_T_dest, start_T_dest + total_T) if i == axis_t else slice(None)
            for i in range(self.ndim)
        )

        # Build compute graph across all levels, then compute once.
        ops = []
        cur = batch_arr
        for level_index in range(self.num_levels):
            if level_index > 0:
                nextshape = list(self.level_shapes[level_index])
                nextshape[axis_t] = total_T
                cur = resize(cur, tuple(nextshape), order=0).astype(cur.dtype)

            # Align to target chunks
            tgt_chunks = self.datasets[level_index].chunks
            src = cur if cur.chunks == tgt_chunks else cur.rechunk(tgt_chunks)

            if self.zarr_format == 2:
                ops.append(
                    da.to_zarr(
                        src,
                        self.datasets[level_index],
                        compute=False,
                        region=region_tuple,
                    )
                )
            else:
                ops.append(
                    da.store(
                        src,
                        self.datasets[level_index],
                        regions=region_tuple,
                        lock=True,
                        compute=False,
                    )
                )
        # compute and let dask optimize
        da.compute(*ops)

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
            for level_index, level_shape in enumerate(self.level_shapes):
                chunks_lvl = self.chunk_shapes_per_level[level_index]
                arr = self.root.zeros(
                    name=str(level_index),
                    shape=level_shape,
                    chunks=chunks_lvl,
                    dtype=self.dtype,
                    compressor=compressor,
                    zarr_format=2,
                    dimension_separator="/",
                )
                self.datasets.append(arr)
        else:
            # v3
            for level_index, level_shape in enumerate(self.level_shapes):
                chunks_lvl = self.chunk_shapes_per_level[level_index]
                kwargs: Dict[str, Any] = {
                    "name": str(level_index),
                    "shape": level_shape,
                    "chunks": chunks_lvl,
                    "dtype": self.dtype,
                    "compressors": compressor,
                    "chunk_key_encoding": {
                        "name": "default",
                        "separator": "/",
                    },
                    "dimension_names": list(self.axes.names),
                }
                # Per-level shards if provided (Zarr v3 only)
                if self.shards_per_level is not None:
                    kwargs["shards"] = tuple(
                        int(x) for x in self.shards_per_level[level_index]
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
