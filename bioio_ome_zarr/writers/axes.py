from typing import List, Optional, Tuple


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
        Physical pixel sizes at level 0 per axis.
    factors : Tuple[int, ...]
        Per-level downsample factor for each axis. Typically >1 for spatial axes.
    """

    # Default NGFF axis definitions (trimmed to match ndim)
    DEFAULT_NAMES: List[str] = ["t", "c", "z", "y", "x"]
    DEFAULT_TYPES: List[str] = ["time", "channel", "space", "space", "space"]
    DEFAULT_UNITS: List[Optional[str]] = [None, None, None, None, None]
    DEFAULT_SCALES: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0]

    def __init__(
        self,
        ndim: int,
        factors: Tuple[int, ...],
        names: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        units: Optional[List[Optional[str]]] = None,
        scales: Optional[List[float]] = None,
    ) -> None:
        """
        Initialize an Axes object.

        Parameters
        ----------
        ndim : int
            Number of dimensions in the image data.
        factors : Tuple[int, ...]
            Per-axis downsample factors.
        names : Optional[List[str]], default=None
            Axis names. Defaults to the last `ndim` entries from DEFAULT_NAMES.
        types : Optional[List[str]], default=None
            Axis types. Defaults to the last `ndim` entries from DEFAULT_TYPES.
        units : Optional[List[Optional[str]]], default=None
            Axis units. Defaults to the last `ndim` entries from DEFAULT_UNITS.
        scales : Optional[List[float]], default=None
            Axis scales. Defaults to the last `ndim` entries from DEFAULT_SCALES.
        """
        self.ndim = ndim
        self.names = names[-ndim:] if names is not None else Axes.DEFAULT_NAMES[-ndim:]
        self.types = types[-ndim:] if types is not None else Axes.DEFAULT_TYPES[-ndim:]
        self.units = units[-ndim:] if units is not None else Axes.DEFAULT_UNITS[-ndim:]
        self.scales = (
            scales[-ndim:] if scales is not None else Axes.DEFAULT_SCALES[-ndim:]
        )
        self.factors = factors[-ndim:]

    def to_metadata(self) -> List[dict]:
        """
        Convert axis info to NGFF-style metadata dicts.

        Returns
        -------
        List[dict]
            List of dictionaries with keys: name, type, and optional unit.
        """
        return [
            {"name": n, "type": t, **({"unit": u} if u is not None else {})}
            for n, t, u in zip(self.names, self.types, self.units)
        ]

    def scales_for_level(self, level: int) -> List[float]:
        """
        Compute the per-axis physical scale at a given pyramid level.

        Spatial axes are scaled by:
            base_scale * (factor ** level)
        Non-spatial axes return 1.0 regardless of level.

        Parameters
        ----------
        level : int
            Pyramid level index (0 = full resolution).

        Returns
        -------
        List[float]
            Per-axis physical scale for the given level.
        """
        out: List[float] = []
        for s, f, t in zip(self.scales, self.factors, self.types):
            if t == "space":
                out.append(float(s) * (float(f) ** float(level)))
            else:
                out.append(1.0)
        return out

    def index_of(self, axis_name: str) -> int:
        """
        Get the index of an axis by name (case-insensitive).

        Parameters
        ----------
        axis_name : str
            Axis name to search for.

        Returns
        -------
        int
            Index of the axis.

        Raises
        ------
        ValueError
            If the axis is not present.
        """
        lowered = [n.lower() for n in self.names]
        if axis_name.lower() not in lowered:
            raise ValueError(f"Axis '{axis_name}' not present: {self.names}")
        return lowered.index(axis_name.lower())
