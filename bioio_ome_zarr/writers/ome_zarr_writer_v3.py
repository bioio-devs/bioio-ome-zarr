import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

from .axes import Axes
from .channel import Channel


def spatial_downsample(
    data: np.ndarray,
    axes_names: List[str],
    scale_factors: Tuple[int, ...],
) -> np.ndarray:
    """
    Downsample the input array along spatial (X/Y) axes.

    Parameters
    ----------
    data : np.ndarray
        Input array of dimensionality N. Only axes named "x" or "y"
        are downsampled.
    axes_names : List[str]
        List of length N containing axis names (e.g.
        ["t", "c", "z", "y", "x"]).
    scale_factors : Tuple[int, ...]
        Tuple of length N containing downsampling factors for each axis.

    Returns
    -------
    np.ndarray
        Array in which each "x" or "y" axis is downsampled by its
        corresponding factor. Non-spatial axes remain unchanged.

    Notes
    -----
    * Crop each spatial axis to a multiple of its factor before
      reshaping.
    * Compute the mean along each factor dimension to downsample.
    """
    names_lower = [n.lower() for n in axes_names]
    shape = data.shape

    crop_slices: List[Union[slice, int]] = []
    reshape_dims: List[int] = []
    factor_axes: List[int] = []
    new_dim_idx = 0

    for i, dim_size in enumerate(shape):
        ax_name = names_lower[i]
        f = scale_factors[i]

        if ax_name in ("x", "y") and f > 1 and dim_size >= f:
            full = (dim_size // f) * f
            crop_slices.append(slice(0, full))
            reshape_dims.extend([full // f, f])
            factor_axes.append(new_dim_idx + 1)
            new_dim_idx += 2
        else:
            crop_slices.append(slice(None))
            reshape_dims.append(dim_size)
            new_dim_idx += 1

    data_cropped = data[tuple(crop_slices)]
    data_reshaped = data_cropped.reshape(tuple(reshape_dims))
    downsampled = data_reshaped.mean(axis=tuple(factor_axes))

    return downsampled


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

    def write_full_volume(self, data: np.ndarray) -> None:
        """
        Write an entire image volume into the multiscale pyramid.

        Parameters
        ----------
        data : np.ndarray
            Full-resolution volume matching self.shape. All levels are
            generated by repeated spatial downsampling.

        Notes
        -----
        * Level 0 is written directly. Subsequent levels use repeated
          calls to spatial_downsample().
        * Requires data.shape == self.shape.
        """
        self.datasets[0][...] = data
        cur = data
        for lvl in range(1, self.num_levels):
            cur = spatial_downsample(cur, self.axes.names, self.axes.factors)
            self.datasets[lvl][...] = cur

    def write_timepoint(self, t_index: int, data_t: np.ndarray) -> None:
        """
        Write a single timepoint slice into the multiscale pyramid.

        Parameters
        ----------
        t_index : int
            Index along the time axis for this slice.
        data_t : np.ndarray
            A single time slice of shape self.shape[1:] (all dims except
            time). Must match expected per-timepoint shape.

        Notes
        -----
        * Finds the "t" axis in self.axes.names and writes data_t into
          level 0 at index t_index.
        * Builds a 1×… slice, downsamples spatial axes via
          spatial_downsample(), and writes each slice into the matching
          time index at coarser levels.
        * Requires that "t" appears in self.axes.names.
        """
        # Find the index of the "t" axis
        axis_t = [i for i, n in enumerate(self.axes.names) if n.lower() == "t"][0]

        # Selector for level 0: List[Union[slice,int]]
        sel0: List[Union[slice, int]] = [slice(None)] * self.ndim
        sel0[axis_t] = t_index  # type: ignore[index]
        self.datasets[0][tuple(sel0)] = data_t

        # Expand dims along time axis for downsampling
        block = np.expand_dims(data_t, axis=axis_t)

        for lvl in range(1, self.num_levels):
            block = spatial_downsample(block, self.axes.names, self.axes.factors)

            # Selector to extract the single frame from the block
            sl_sngl: List[Union[slice, int]] = [slice(None)] * self.ndim
            sl_sngl[axis_t] = 0  # type: ignore[index]
            slice_data = block[tuple(sl_sngl)]

            # Selector to place that slice at time index t_index
            sel_lvl: List[Union[slice, int]] = [slice(None)] * self.ndim
            sel_lvl[axis_t] = t_index  # type: ignore[index]
            self.datasets[lvl][tuple(sel_lvl)] = slice_data
