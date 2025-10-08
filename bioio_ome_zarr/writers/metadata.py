from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Public NGFF version constants
OME_NGFF_VERSION_V04 = "0.4"
OME_NGFF_VERSION_V05 = "0.5"


class Axes:
    """
    Holds axis metadata for an N-D image, aligned with NGFF 0.5 axes, and
    renders cleanly for both NGFF 0.4 (Zarr v2) and NGFF 0.5 (Zarr v3).

    Attributes
    ----------
    ndim : int
        Number of dimensions in the image data.
    names : List[str]
        Names of each axis (e.g., ["t", "c", "z", "y", "x"]).
    types : List[str]
        NGFF axis types for each axis (e.g., "time", "channel", "space").
    units : List[Optional[str]]
        Physical units for each axis, if any (e.g., "micrometer").
    scales : List[float]
        Physical scale at level 0 per axis (e.g., pixel sizes).
    factors : Tuple[int, ...]
        Per-axis downsample factor (not required by NGFF; useful for clients).
    """

    DEFAULT_NAMES: List[str] = ["t", "c", "z", "y", "x"]
    DEFAULT_TYPES: List[str] = ["time", "channel", "space", "space", "space"]
    DEFAULT_UNITS: List[Optional[str]] = [None, None, None, None, None]
    DEFAULT_SCALES: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0]

    def __init__(
        self,
        *,
        ndim: int,
        names: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        units: Optional[List[Optional[str]]] = None,
        scales: Optional[List[float]] = None,
        factors: Tuple[int, ...] = (),
    ) -> None:
        self.ndim = int(ndim)
        self.names = list(
            names[-ndim:] if names is not None else self.DEFAULT_NAMES[-ndim:]
        )
        self.types = list(
            types[-ndim:] if types is not None else self.DEFAULT_TYPES[-ndim:]
        )
        self.units = list(
            units[-ndim:] if units is not None else self.DEFAULT_UNITS[-ndim:]
        )
        self.scales = list(
            scales[-ndim:] if scales is not None else self.DEFAULT_SCALES[-ndim:]
        )
        self.factors = tuple(factors[-ndim:] if factors else (1,) * self.ndim)

        if not (
            len(self.names)
            == len(self.types)
            == len(self.units)
            == len(self.scales)
            == self.ndim
        ):
            raise ValueError("Axes fields must all match ndim.")

    def to_metadata(self) -> List[Dict[str, Any]]:
        """Return NGFF-style axis dicts with keys: name, type, and optional unit."""
        out: List[Dict[str, Any]] = []
        for n, t, u in zip(self.names, self.types, self.units):
            d: Dict[str, Any] = {"name": n, "type": t}
            if u is not None:
                d["unit"] = u
            out.append(d)
        return out

    def index_of(self, axis_name: str) -> int:
        """Case-insensitive index lookup for an axis."""
        lowered = [n.lower() for n in self.names]
        key = axis_name.lower()
        if key not in lowered:
            raise ValueError(f"Axis '{axis_name}' not present in {self.names}")
        return lowered.index(key)


class Channel:
    """
    Helper to construct an OMERO-style channel dict compliant with
    the NGFF OME-Zarr 0.5 OMERO block:
    https://ngff.openmicroscopy.org/0.5/#omero-md

    Only `label` and `color` are required; others have sensible defaults.
    `window` defaults to 0–255.
    """

    def __init__(
        self,
        *,
        label: str,
        color: str,
        active: bool = True,
        coefficient: float = 1.0,
        family: str = "linear",
        inverted: bool = False,
        window: Optional[Dict[str, int]] = None,
    ) -> None:
        self.label = label
        self.color = color
        self.active = bool(active)
        self.coefficient = float(coefficient)
        self.family = family
        self.inverted = bool(inverted)
        self.window = window or {"min": 0, "max": 255, "start": 0, "end": 255}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color": self.color,
            "coefficient": self.coefficient,
            "active": self.active,
            "label": self.label,
            "window": {
                "min": self.window["min"],
                "max": self.window["max"],
                "start": self.window["start"],
                "end": self.window["end"],
            },
            "family": self.family,
            "inverted": self.inverted,
        }


@dataclass
class MetadataParams:
    """
    Parameters required to render NGFF metadata.

    Fields
    ------
    image_name : str
        Display name of the image.
    axes : Axes
        Axes descriptor (names/types/units/scales).
    level_shapes : Sequence[Tuple[int, ...]]
        Per-level array shapes, including level 0.
    channels : Optional[List[Channel]]
        OMERO-style channel metadata; if None, defaults are inferred.
    rdefs : Optional[Dict[str, Any]]
        OMERO rendering defaults (placed under ome->omero->rdefs for 0.5).
    creator_info : Optional[Dict[str, Any]]
        Optional creator block stored under ome->_creator (0.5).
    root_transform : Optional[Dict[str, Any]]
        Optional transform at the multiscale root.
    dataset_scales : Optional[List[List[float]]]
        For levels > 0, per-axis *relative size vs. level 0*.
        Example: for level 1 where spatial dims halve, use [1,1,1,0.5,0.5].
        If None, only level 0 is expected.
    """

    image_name: str
    axes: Axes
    level_shapes: Sequence[Tuple[int, ...]]
    channels: Optional[List[Channel]] = None
    rdefs: Optional[Dict[str, Any]] = None
    creator_info: Optional[Dict[str, Any]] = None
    root_transform: Optional[Dict[str, Any]] = None
    dataset_scales: Optional[List[List[float]]] = None


def build_ngff_metadata(
    *,
    zarr_format: int,
    params: MetadataParams,
) -> Dict[str, Any]:
    """
    Build NGFF metadata dicts for the given zarr_format.

    Returns
    -------
    Dict[str, Any]
        For zarr_format == 2: {"multiscales": [...], "omero": {...}}
        For zarr_format == 3: {"ome": {...}}
    """
    if zarr_format == 2:
        multiscales, omero = _build_ngff_v04(params)
        return {"multiscales": multiscales, "omero": omero}
    else:
        ome_block = _build_ngff_v05(params)
        return {"ome": ome_block}


# ----------------------------------------------------------------------
# Internal helpers (pure build-time logic)
# ----------------------------------------------------------------------


def _build_ngff_v04(p: MetadataParams) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build NGFF 0.4 `multiscales` and `omero` dicts (Zarr v2 layout)."""
    axes_list = p.axes.to_metadata()

    datasets: List[Dict[str, Any]] = []
    for lvl in range(len(p.level_shapes)):
        scale_vec: List[float] = []
        for i in range(len(p.axes.names)):
            base = float(p.axes.scales[i] if i < len(p.axes.scales) else 1.0)
            if lvl == 0 or p.dataset_scales is None:
                scale_vec.append(base)
            else:
                rel_size = float(p.dataset_scales[lvl - 1][i])
                if rel_size == 0.0:
                    rel_size = 1.0
                scale_vec.append(base * (1.0 / rel_size))

        datasets.append(
            {
                "path": str(lvl),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": scale_vec,
                    }
                ],
            }
        )

    multiscale: Dict[str, Any] = {
        "axes": axes_list,
        "datasets": datasets,
        "name": p.image_name,
        "version": OME_NGFF_VERSION_V04,
    }

    # Only include root transform if explicitly provided
    if p.root_transform is not None:
        multiscale["coordinateTransformations"] = [p.root_transform]

    multiscales = [multiscale]

    # OMERO channels: provided or inferred
    if p.channels:
        channel_list = [ch.to_dict() for ch in p.channels]
    else:
        try:
            c_axis = p.axes.index_of("c")
            C = int(p.level_shapes[0][c_axis])
        except ValueError:
            C = 1
        channel_list = [
            Channel(label=f"C:{i}", color="ffffff").to_dict() for i in range(C)
        ]

    # defaultZ for rdefs
    try:
        z_axis = p.axes.index_of("z")
        size_z = int(p.level_shapes[0][z_axis])
    except ValueError:
        size_z = 1

    omero = {
        "id": 1,
        "name": p.image_name,
        "version": OME_NGFF_VERSION_V04,
        "channels": channel_list,
        "rdefs": {"defaultT": 0, "defaultZ": max(0, size_z // 2), "model": "color"},
    }
    return multiscales, omero


def _build_ngff_v05(p: MetadataParams) -> Dict[str, Any]:
    """Build NGFF 0.5 `ome` dict with multiscales and optional OMERO (Zarr v3)."""
    axes_list = p.axes.to_metadata()

    datasets: List[Dict[str, Any]] = []
    for lvl in range(len(p.level_shapes)):
        scale_vec: List[float] = []
        for i in range(len(p.axes.names)):
            base = float(p.axes.scales[i] if i < len(p.axes.scales) else 1.0)
            if lvl == 0 or p.dataset_scales is None:
                scale_vec.append(base)
            else:
                rel_size = float(p.dataset_scales[lvl - 1][i])
                if rel_size == 0.0:
                    rel_size = 1.0
                scale_vec.append(base * (1.0 / rel_size))

        datasets.append(
            {
                "path": str(lvl),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": scale_vec,
                    }
                ],
            }
        )

    multiscale: Dict[str, Any] = {
        "name": p.image_name,
        "axes": axes_list,
        "datasets": datasets,
    }

    # Only include root transform if explicitly provided
    if p.root_transform is not None:
        multiscale["coordinateTransformations"] = [p.root_transform]

    ome: Dict[str, Any] = {"version": OME_NGFF_VERSION_V05, "multiscales": [multiscale]}

    # Optional OMERO block
    if p.channels:
        ome["omero"] = {
            "version": OME_NGFF_VERSION_V05,
            "channels": [ch.to_dict() for ch in p.channels],
        }
        if p.rdefs is not None:
            ome["omero"]["rdefs"] = p.rdefs

    if p.creator_info:
        ome["_creator"] = p.creator_info

    return ome
