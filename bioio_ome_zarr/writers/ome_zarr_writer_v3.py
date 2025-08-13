from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import numcodecs
import numpy as np
import zarr
from numcodecs import Blosc as BloscV2
from zarr.codecs import BloscCodec, BloscShuffle

from .axes import Axes
from .channel import Channel
from .metadata import MetadataParams, write_ngff_metadata
from .utils import (
    chunk_size_from_memory_target,
    compute_level_chunk_sizes_zslice,
    compute_level_shapes,
    resize,
)


class OMEZarrWriterV3:
    """
    OMEZarrWriterV3 is a unified OME‑Zarr writer that targets either Zarr v2
    (NGFF 0.4) or Zarr v3 (NGFF 0.5) with the same public API. Supports
    2 ≤ N ≤ 5 dimensions (e.g., YX, ZYX, TYX, CZYX, or TCZYX) and writes a
    multiscale pyramid with nearest‑neighbor downsampling.
    """

    # -----------------------
    # Small dataclass (levels)
    # -----------------------
    @dataclass
    class ZarrLevel:
        shape: Tuple[int, ...]
        chunk_size: Tuple[int, ...]
        dtype: np.dtype
        zarray: zarr.Array

    def __init__(
        self,
        store: Union[str, zarr.storage.StoreLike],
        shape: Tuple[int, ...],
        dtype: Union[np.dtype, str],
        scale_factors: Tuple[int, ...],
        axes_names: Optional[List[str]] = None,
        axes_types: Optional[List[str]] = None,
        axes_units: Optional[List[Optional[str]]] = None,
        axes_scale: Optional[List[float]] = None,
        num_levels: Optional[int] = None,
        chunk_size: Optional[Tuple[int, ...]] = None,
        shard_factor: Optional[Tuple[int, ...]] = None,
        compressor: Optional[Union[BloscCodec, numcodecs.abc.Codec]] = None,
        image_name: str = "Image",
        # NGFF/OMERO optional metadata
        channels: Optional[List[Channel]] = None,
        rdefs: Optional[dict] = None,
        creator_info: Optional[dict] = None,
        root_transform: Optional[Dict[str, Any]] = None,
        *,
        zarr_format: Literal[2, 3] = 3,
        ngff_version: Optional[str] = None,
    ) -> None:
        """
        Initialize the writer and capture core configuration. Arrays and
        metadata are created lazily on the first write.

        Parameters
        ----------
        store : Union[str, zarr.storage.StoreLike]
            Filesystem path, URL (via fsspec), or Store-like for the root group.
        shape : Tuple[int, ...]
            Level‑0 image shape (e.g., (T,C,Z,Y,X)).
        dtype : Union[np.dtype, str]
            NumPy dtype for the on-disk array.
        scale_factors : Tuple[int, ...]
            Integer downsample factor per axis; typically >1 for spatial axes.
        axes_names : Optional[List[str]]
            Axis names; defaults to last N of ["t","c","z","y","x"].
        axes_types : Optional[List[str]]
            Axis types; defaults to ["time","channel","space",...].
        axes_units : Optional[List[Optional[str]]]
            Physical units for each axis.
        axes_scale : Optional[List[float]]
            Physical scale at level 0 for each axis.
        num_levels : Optional[int]
            Number of pyramid levels; if None, computed by utility.
        chunk_size : Optional[Tuple[int, ...]]
            Chunk size for arrays (uniform for v3). If None, auto‑sized≈16 MiB.
        shard_factor : Optional[Tuple[int, ...]]
            Optional shard factor for v3 arrays; ignored for v2.
        compressor : Optional[BloscCodec | numcodecs.abc.Codec]
            Codec to use. For v2 use numcodecs.Blosc; for v3 use zarr.codecs.BloscCodec.
        image_name : str
            Image name used in multiscales metadata.
        channels : Optional[List[Channel]]
            OMERO‑style channel metadata objects. If None, defaults are inferred.
        rdefs : Optional[dict]
            Optional OMERO rendering defaults.
        creator_info : Optional[dict]
            Optional creator block placed in metadata (v0.5).
        root_transform : Optional[Dict[str, Any]]
            Optional multiscale root coordinate transformation.
        zarr_format : Literal[2,3]
            Target Zarr array format.
        ngff_version : Optional[str]
            Informational version tag to record alongside metadata.
        """
        # 1) Store fundamental properties
        self.store = store
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)

        # 2) Build an Axes instance (handles defaults internally)
        self.axes = Axes(
            ndim=self.ndim,
            names=axes_names,
            types=axes_types,
            units=axes_units,
            scales=axes_scale,
            factors=scale_factors,
        )

        # 3) Compute all pyramid level shapes
        self.level_shapes = compute_level_shapes(
            self.shape, self.axes.names, self.axes.factors, num_levels
        )
        self.num_levels = len(self.level_shapes)

        # 4) Determine chunk size (uniform for v3; v2 per‑level in _initialize)
        if chunk_size is None:
            self.chunk_size = chunk_size_from_memory_target(
                self.level_shapes[0], self.dtype, 16 << 20
            )
        else:
            self.chunk_size = chunk_size

        # 5) Layout & codec preferences
        self.zarr_format = zarr_format
        self.shard_factor = shard_factor  # ignored for v2 arrays
        self.compressor = compressor

        # 6) Metadata fields (passed to metadata builder later)
        self.image_name = image_name
        self.channels = channels
        self.rdefs = rdefs
        self.creator_info = creator_info
        self.root_transform = root_transform

        # 7) Informational NGFF version selection
        self.ngff_version = ngff_version or ("0.4" if zarr_format == 2 else "0.5")

        # 8) Handles & state
        self.root: Optional[zarr.Group] = None
        self.datasets: List[zarr.Array] = []
        self.levels: List[OMEZarrWriterV3.ZarrLevel] = []
        self._initialized: bool = False
        self._metadata_written: bool = False

    # -----------------
    # Public interface
    # -----------------

    def write_full_volume(self, input_data: Union[np.ndarray, da.Array]) -> None:
        """
        Write full‑resolution data into all pyramid levels.

        Parameters
        ----------
        input_data : Union[np.ndarray, dask.array.Array]
            Array matching level‑0 shape. If NumPy, it will be wrapped into a
            Dask array with level‑0 chunking.
        """
        if not self._initialized:
            self._initialize()

        base = (
            input_data
            if isinstance(input_data, da.Array)
            else da.from_array(input_data, chunks=self.datasets[0].chunks)
        )

        # Store each level (downsampled with nearest‑neighbor for parity)
        for lvl, shape in enumerate(self.level_shapes):
            src = base if lvl == 0 else resize(base, shape, order=0)
            if self.zarr_format == 2:
                da.to_zarr(src, self.datasets[lvl])
            else:
                da.store(src, self.datasets[lvl], lock=True)

    def write_timepoint(
        self, t_index: int, data_t: Union[np.ndarray, da.Array]
    ) -> None:
        """
        Write a single timepoint slice at index ``t_index`` into all levels.

        Parameters
        ----------
        t_index : int
            Index along the time axis to write.
        data_t : Union[np.ndarray, dask.array.Array]
            One timepoint with shape equal to level‑0 shape **without** the T axis.
        """
        if not self._initialized:
            self._initialize()

        # Locate the time axis (raises if absent)
        try:
            axis_t = self.axes.index_of("t")
        except ValueError:
            raise ValueError("Axes do not include a time axis 't'.")

        # Ensure a Dask array with a singleton T dimension
        if isinstance(data_t, da.Array):
            block = da.expand_dims(data_t, axis=axis_t)
        else:
            level0_chunks = self.datasets[0].chunks
            chunks_wo_t = level0_chunks[:axis_t] + level0_chunks[axis_t + 1 :]
            block = da.expand_dims(
                da.from_array(data_t, chunks=chunks_wo_t), axis=axis_t
            )

        # Downsample & store this timepoint per level
        for lvl in range(self.num_levels):
            level_shape = (1,) + self.level_shapes[lvl][1:]
            level_block = block if lvl == 0 else resize(block, level_shape, order=0)

            if self.zarr_format == 2:
                # Use region write to materialize only this timepoint
                sel_region: List[slice] = [slice(None)] * self.ndim
                sel_region[axis_t] = slice(t_index, t_index + 1)
                da.to_zarr(level_block, self.datasets[lvl], region=tuple(sel_region))
            else:
                arr = level_block.compute()[0]
                from typing import cast

                sel_set: List[Union[slice, int]] = [
                    cast(Union[slice, int], slice(None)) for _ in range(self.ndim)
                ]
                sel_set[axis_t] = t_index
                self.datasets[lvl][tuple(sel_set)] = arr

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
                compressor = BloscV2(cname="zstd", clevel=3, shuffle=BloscV2.BITSHUFFLE)
            else:
                compressor = BloscCodec(
                    cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle
                )
        else:
            compressor = self.compressor

        self.datasets = []
        self.levels = []

        if self.zarr_format == 2:
            # v2
            level_chunk_sizes = compute_level_chunk_sizes_zslice(self.level_shapes)
            for lvl, shape in enumerate(self.level_shapes):
                chunks_lvl = level_chunk_sizes[lvl]
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
                self.levels.append(
                    OMEZarrWriterV3.ZarrLevel(
                        shape=shape, chunk_size=chunks_lvl, dtype=self.dtype, zarray=arr
                    )
                )
        else:
            # v3
            for lvl, shape in enumerate(self.level_shapes):
                kwargs: Dict[str, Any] = {
                    "name": str(lvl),
                    "shape": shape,
                    "chunks": self.chunk_size,
                    "dtype": self.dtype,
                    "compressors": compressor,
                    "chunk_key_encoding": {"name": "default", "separator": "/"},
                }
                if self.shard_factor is not None:
                    kwargs["shards"] = tuple(
                        c * f for c, f in zip(self.chunk_size, self.shard_factor)
                    )
                arr = self.root.create_array(**kwargs)
                self.datasets.append(arr)
                self.levels.append(
                    OMEZarrWriterV3.ZarrLevel(
                        shape=shape,
                        chunk_size=self.chunk_size,
                        dtype=self.dtype,
                        zarray=arr,
                    )
                )

        # Write multiscale + OMERO metadata
        self._write_metadata_according_to_format()
        self._initialized = True
        self._metadata_written = True

    def _open_root(self) -> zarr.Group:
        """Accept a path/URL or Store-like and return an opened root group."""
        if isinstance(self.store, str):
            if "://" in self.store:
                fs = zarr.storage.FsspecStore(self.store, mode="w")
                return zarr.open_group(store=fs, mode="w", zarr_format=self.zarr_format)
            return zarr.open_group(self.store, mode="w", zarr_format=self.zarr_format)
        return zarr.group(
            store=self.store, overwrite=True, zarr_format=self.zarr_format
        )

    def _write_metadata_according_to_format(self) -> None:
        """Build and persist NGFF metadata (v0.4 for v2, v0.5 for v3)."""
        if self.root is None:
            raise RuntimeError("Store must be initialized before writing metadata.")

        params = MetadataParams(
            image_name=self.image_name,
            axes=self.axes,
            level_shapes=self.level_shapes,
            channels=self.channels,
            rdefs=self.rdefs,
            creator_info=self.creator_info,
            root_transform=self.root_transform,
        )
        write_ngff_metadata(self.root, zarr_format=self.zarr_format, params=params)
