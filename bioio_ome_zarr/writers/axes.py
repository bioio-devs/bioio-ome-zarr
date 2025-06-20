from typing import List, Optional, Tuple


class Axes:
    """
    Holds axis metadata for an N-D image, in accordance with
    the NGFF OME-Zarr 0.5.0 metadata specification for axes
    (see https://ngff.openmicroscopy.org/0.5/#axes-md).

    After initialization:
      - names   is length N
      - types   is length N
      - units   is length N
      - scales  is length N
      - factors is length N
    """

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
    ):
        self.ndim = ndim
        self.names = names[-ndim:] if names is not None else Axes.DEFAULT_NAMES[-ndim:]
        self.types = types[-ndim:] if types is not None else Axes.DEFAULT_TYPES[-ndim:]
        self.units = units[-ndim:] if units is not None else Axes.DEFAULT_UNITS[-ndim:]
        self.scales = (
            scales[-ndim:] if scales is not None else Axes.DEFAULT_SCALES[-ndim:]
        )
        self.factors = factors[-ndim:]

    def to_metadata(self) -> List[dict]:
        return [
            {"name": n, "type": t, **({"unit": u} if u is not None else {})}
            for n, t, u in zip(self.names, self.types, self.units)
        ]
