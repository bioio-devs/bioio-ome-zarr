import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

from .axes import Axes
from .channel import Channel
from .ome_zarr_writer_v2 import resize


class OMEZarrWriterV3:
    """
    OMEZarrWriterV3 is a fully compliant OME-Zarr v0.5.0 writer built
    on Zarr v3 stores. Supports 2 ≤ N ≤ 5 dimensions (e.g. YX, ZYX,
    TYX, CZYX, or TCZYX).
    """

    def __init__(
        self,
        store: Union[str, zarr.storage.StoreLike],
        shape: Tuple[int, ...],
        dtype: Union[np.dtype, str],
        axes_names: Optional[List[str]] = None,
        axes_types: Optional[List[str]] = None,
        axes_units: Optional[List[Optional[str]]] = None,
        axes_scale: Optional[List[float]] = None,
        scale_factors: Optional[Tuple[int, ...]] = None,
        num_levels: Optional[int] = None,
        chunks: Union[str, Tuple[int, ...], List[Tuple[int, ...]]] = "auto",
        shards: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
        compressor: Optional[BloscCodec] = None,
        image_name: str = "Image",
        channels: Optional[List[Channel]] = None,
        rdefs: Optional[dict] = None,
        creator_info: Optional[dict] = None,
        multiscale_scale: Optional[List[float]] = None,
    ):
        """
        Initialize writer and build axes + channel metadata automatically.

        Parameters
        ----------
        store : Union[str, zarr.storage.StoreLike]
            Path or Zarr store-like object for the output group.
        shape : Tuple[int, ...]
            Base image shape (e.g. (Y, X), (Z, Y, X), (T, C, Z, Y, X)).
        dtype : Union[np.dtype, str]
            NumPy dtype of the image data (e.g. "uint8").
        axes_names : Optional[List[str]]
            Names of each axis, len = len(shape). Defaults to last N of
            ["t", "c", "z", "y", "x"].
        axes_types : Optional[List[str]]
            Types of each axis (e.g. ["time", "channel", "space"...]).
        axes_units : Optional[List[Optional[str]]]
            Physical units for each axis (e.g. ["ms", None, "µm"]).
        axes_scale : Optional[List[float]]
            Physical scale per axis at base resolution.
        scale_factors : Optional[Tuple[int, ...]]
            Integer downsampling factors per axis (e.g. (1, 1, 2, 2)).
        num_levels : Optional[int]
            Number of pyramid levels to generate. If None, compute until
            no further reduction.
        chunks : Union[str, Tuple[int, ...], List[Tuple[int, ...]]]
            Chunk sizes: "auto" or a tuple or list per level.
        shards : Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]
            Shard factors per level. None for no sharding.
        compressor : Optional[BloscCodec]
            Zarr compressor to use (default: Blosc Zstd).
        image_name : str
            Name for the image in multiscales metadata.
        channels : Optional[List[Channel]]
            OMERO-style channel metadata objects.
        rdefs : Optional[dict]
            OMERO rendering settings (passed under "omero" → "rdefs").
        creator_info : Optional[dict]
            Creator metadata dictionary (e.g.
            {"name":"pytest","version":"0.1"}).
        multiscale_scale : Optional[List[float]]
            Optional top-level scale transform (list of floats). Inserted
            under "coordinateTransformations" of the multiscale block.
        """
        # 1) Store fundamental properties
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
        self.level_shapes = self._compute_levels(num_levels)

        # 4) Record actual number of levels
        self.num_levels = len(self.level_shapes)

        # 5) Prepare chunk & shard parameters per level
        self.chunks = self._prepare_parameter(chunks, self._suggest_chunks, "chunks")
        self.shards = self._prepare_parameter(
            shards, lambda s: s, "shards", required=False
        )

        # 6) Initialize Zarr store and create arrays for each level
        self.root = self._init_store(store)
        self.datasets = self._create_arrays(compressor)

        # 7) Convert Channel objects to raw dicts (if provided)
        if channels is not None:
            self.channels = [ch.to_dict() for ch in channels]
        else:
            self.channels: Optional[List[Dict[str, Any]]] = None  # type: ignore

        # 8) Store OMERO render settings verbatim
        self.rdefs = rdefs

        # 9) Capture optional top-level scale transform
        self.multiscale_scale = multiscale_scale

        # 10) Build axes metadata list
        axes_meta = self.axes.to_metadata()

        # 11) Write the OME-Zarr metadata block
        self._write_metadata(
            name=image_name,
            axes=axes_meta,
            channels=self.channels,
            rdefs=self.rdefs,
            creator=creator_info,
        )

    def _compute_levels(self, max_levels: Optional[int]) -> List[Tuple[int, ...]]:
        """
        Compute pyramid level shapes by repeatedly downsampling spatial axes.

        Parameters
        ----------
        max_levels : Optional[int]
            Maximum number of levels (including base). If None, generate
            until downsampled shape equals previous.

        Returns
        -------
        List[Tuple[int, ...]]
            List of shape tuples for each level.

        Notes
        -----
        * Only axes named "x" or "y" get downsampled by `scale_factors`.
        * Stops early if new shape equals the last shape.
        """
        shapes: List[Tuple[int, ...]] = [self.shape]
        lvl = 1

        while max_levels is None or lvl < max_levels:
            prev = shapes[-1]
            nxt: List[int] = []
            for i, size in enumerate(prev):
                ax_name = self.axes.names[i].lower()
                factor = self.axes.factors[i]
                if ax_name in ("x", "y") and factor > 1:
                    nxt.append(max(1, size // factor))
                else:
                    nxt.append(size)
            nxt_tuple = tuple(nxt)
            if nxt_tuple == prev:
                break
            shapes.append(nxt_tuple)
            lvl += 1

        return shapes

    def _prepare_parameter(
        self,
        param: Any,
        default_fn: Callable[[Tuple[int, ...]], Tuple[int, ...]],
        name: str,
        required: bool = True,
    ) -> List[Optional[Tuple[int, ...]]]:
        """
        Standardize chunk or shard specification across levels.

        Parameters
        ----------
        param : Any
            Either "auto", None, a single tuple, or a list of tuples.
        default_fn : Callable[[Tuple[int, ...]], Tuple[int, ...]]
            Function to generate a default chunk size for a given shape.
        name : str
            "chunks" or "shards".
        required : bool
            If False and param is None, returns [None] * num_levels.

        Returns
        -------
        List[Optional[Tuple[int, ...]]]
            List of tuples per level or [None,...] if not required.

        Notes
        -----
        * "auto" + name="chunks" → apply default_fn to each level’s shape.
        * None + required=True → apply default_fn on base level and replicate.
        * If param is a tuple or list, ensure length matches num_levels.
        """
        levels = len(self.level_shapes)

        if param == "auto" and name == "chunks":
            return [default_fn(s) for s in self.level_shapes]

        if param is None:
            if not required:
                return [None] * levels
            default = default_fn(self.level_shapes[0])
            return [default] * levels

        if isinstance(param, list):
            items = param
        else:
            items = [param] * levels

        return [
            tuple(min(items[i][d], self.level_shapes[i][d]) for d in range(self.ndim))
            for i in range(levels)
        ]

    @staticmethod
    def _init_store(store: Union[str, zarr.storage.StoreLike]) -> zarr.Group:
        """
        Create or open a Zarr group at the given store location.

        Parameters
        ----------
        store : Union[str, zarr.storage.StoreLike]
            If string containing "://", opens fsspec store in overwrite mode;
            otherwise opens a local group in overwrite mode.

        Returns
        -------
        zarr.Group
            The root group at 'store'.
        """
        if isinstance(store, str) and "://" in store:
            fs = zarr.storage.FsspecStore(store, mode="w")
            return zarr.group(store=fs, overwrite=True)
        return zarr.group(store=store, overwrite=True)

    def _create_arrays(self, resolver: Optional[BloscCodec]) -> List[zarr.Array]:
        """
        Create Zarr arrays for each multiscale level under the root group.

        Parameters
        ----------
        resolver : Optional[BloscCodec]
            If provided, uses this Blosc codec; otherwise uses default Zstd.

        Returns
        -------
        List[zarr.Array]
            One Zarr array per level, named "0", "1", etc.
        """
        comp = resolver or BloscCodec(
            cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle
        )
        arrays: List[zarr.Array] = []
        for lvl, shape in enumerate(self.level_shapes):
            chunks_lvl = self.chunks[lvl]
            shards_lvl = self.shards[lvl]
            if chunks_lvl is not None and shards_lvl is not None:
                shards_param = tuple(c * s for c, s in zip(chunks_lvl, shards_lvl))
            else:
                shards_param = None

            arr = self.root.create_array(
                name=str(lvl),
                shape=shape,
                chunks=chunks_lvl,
                shards=shards_param,
                dtype=self.dtype,
                compressors=comp,
            )
            arrays.append(arr)
        return arrays

    def _write_metadata(
        self,
        name: str,
        axes: List[dict],
        channels: Optional[List[dict]],
        rdefs: Optional[dict],
        creator: Optional[dict],
    ) -> None:
        """
        Write the 'ome' attribute with NGFF v0.5 metadata.

        Parameters
        ----------
        name : str
            Image name stored under "multiscales"[0]["name"].
        axes : List[dict]
            Axis metadata from self.axes.to_metadata().
        channels : Optional[List[dict]]
            OMERO channel metadata, under "omero" → "channels".
        rdefs : Optional[dict]
            OMERO render settings, under "omero" → "rdefs".
        creator : Optional[dict]
            Creator metadata, stored under "_creator".

        Notes
        -----
        * Builds "datasets" list, each with a per-level scale transform.
        * If self.multiscale_scale is provided, inserts it under
          "coordinateTransformations" of the multiscale entry.
        * Updates self.root.attrs["ome"] with the assembled block.
        """
        # Build per-level datasets list
        datasets_list: List[dict] = []
        for lvl in range(self.num_levels):
            scale_vals: List[float] = []
            for ax_i in range(self.ndim):
                if self.axes.types[ax_i] == "space":
                    scale_vals.append(
                        self.axes.scales[ax_i] * (self.axes.factors[ax_i] ** lvl)
                    )
                else:
                    scale_vals.append(1.0)
            datasets_list.append(
                {
                    "path": str(lvl),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale_vals}
                    ],
                }
            )

        # Build the multiscale entry
        multiscale_entry: Dict[str, Any] = {"name": name or ""}

        # Insert top-level scale transform if provided
        if self.multiscale_scale is not None:
            multiscale_entry["coordinateTransformations"] = [
                {"type": "scale", "scale": self.multiscale_scale}
            ]

        multiscale_entry["axes"] = axes
        multiscale_entry["datasets"] = datasets_list

        ome_block: Dict[str, Any] = {
            "version": "0.5",
            "multiscales": [multiscale_entry],
        }

        if channels is not None:
            omero_block: Dict[str, Any] = {"version": "0.5", "channels": channels}
            if rdefs is not None:
                omero_block["rdefs"] = rdefs
            ome_block["omero"] = omero_block

        if creator:
            ome_block["_creator"] = creator

        self.root.attrs.update({"ome": ome_block})

    def _suggest_chunks(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Suggest chunk shapes aiming for ~64MB per chunk.

        Only "space" axes get larger chunks; non-"space" axes → 1.

        Parameters
        ----------
        shape : Tuple[int, ...]
            The shape of the array at the current level.

        Returns
        -------
        Tuple[int, ...]
            A tuple of chunk sizes for each axis.

        Notes
        -----
        * Handles cases where there are exactly 2 or 3 spatial axes.
        * Allocates roughly sqrt(max elements) along X first, then Y,
          then Z.
        """
        bpe = self.dtype.itemsize
        maxe = (64 << 20) // bpe  # max elements ≈ 64MB

        spatial_idxs = [i for i, t in enumerate(self.axes.types) if t == "space"]
        chunk = [1] * self.ndim

        if len(spatial_idxs) in (2, 3):
            remaining = maxe
            base = int(math.sqrt(maxe))
            first = True

            # Assign X, then Y, then Z (if present)
            for idx in reversed(spatial_idxs):
                size = shape[idx]
                if first:
                    val = min(size, base)
                    first = False
                else:
                    val = min(size, max(1, remaining))
                chunk[idx] = val
                remaining //= val

        return tuple(chunk)

    def write_full_volume(self, data: Union[np.ndarray, da.Array]) -> None:
        """
        Write an entire image volume into the multiscale pyramid using Dask or
        NumPy input. Requires reading the full image into memory.

        Parameters
        ----------
        data : Union[np.ndarray, dask.array.Array]
            Full-resolution volume matching self.shape
        Notes
        -----
        * If `data` is a Dask array, uses it directly; otherwise wraps the NumPy
          array with chunking at level-0 shapes.
        * Uses `dask.array.store` to persist into the pre-created Zarr arrays,
          preserving chunk and shard settings.
        """
        # Wrap or reuse input as a Dask array
        if isinstance(data, da.Array):
            darr = data
        else:
            chunk0 = self.chunks[0]
            assert chunk0 is not None
            darr = da.from_array(data, chunks=chunk0)

        # Store each pyramid level
        for lvl, target_shape in enumerate(self.level_shapes):
            if lvl > 0:
                darr = resize(darr, target_shape)
            da.store(darr, self.datasets[lvl], lock=True)

    def write_timepoint(
        self,
        t_index: int,
        data_t: Union[np.ndarray, da.Array],
    ) -> None:
        """
        Write a single timepoint slice into the multiscale pyramid using Dask
        or NumPy input.

        Parameters
        ----------
        t_index : int
            Index along the time axis for this slice.
        data_t : Union[np.ndarray, dask.array.Array]
            A single timepoint slice of shape self.shape[1:].

        Notes
        -----
        * Finds the "t" axis in self.axes.names.
        * Preserves original chunk and shard settings on each Zarr array.
        """
        from typing import List, Union

        # Locate the time axis
        axis_t = next(i for i, n in enumerate(self.axes.names) if n.lower() == "t")

        # Prepare chunk tuple for spatial dims
        chunk0 = self.chunks[0]
        assert chunk0 is not None
        spatial_chunks = chunk0[1:]

        # Wrap or reuse input as a Dask array with time axis
        if isinstance(data_t, da.Array):
            block = da.expand_dims(data_t, axis=axis_t)
        else:
            block = da.expand_dims(
                da.from_array(data_t, chunks=spatial_chunks),
                axis=axis_t,
            )

        # Compute and assign this slice and drop time axis
        arr0 = block.compute()[0]
        sel0: List[Union[slice, int]] = [slice(None)] * self.ndim
        sel0[axis_t] = t_index
        self.datasets[0][tuple(sel0)] = arr0

        for lvl in range(1, self.num_levels):
            # build target shape with time dim = 1
            tgt = list(self.level_shapes[lvl])
            tgt[axis_t] = 1

            # resize returns a Dask array of shape (1, ... spatial dims)
            block = resize(block, tuple(tgt))

            # compute only this downsampled slice and drop T
            arr = block.compute()[0]

            # assign the Zarr array at the time index
            sel: List[Union[slice, int]] = [slice(None)] * self.ndim
            sel[axis_t] = t_index
            self.datasets[lvl][tuple(sel)] = arr
