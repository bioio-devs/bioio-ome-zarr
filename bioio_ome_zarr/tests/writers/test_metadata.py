from typing import Any, Dict, List, Optional

import pytest

from bioio_ome_zarr.writers.metadata import Axes


@pytest.mark.parametrize(
    "ndim, kwargs, expected_types",
    [
        # No names: canonical suffix defaults are preserved.
        (5, {}, ["time", "channel", "space", "space", "space"]),
        (3, {}, ["space", "space", "space"]),
        # Names given: type is inferred per-name, independent of order.
        (
            5,
            {"names": ["t", "c", "z", "y", "x"]},
            ["time", "channel", "space", "space", "space"],
        ),
        (4, {"names": ["c", "z", "y", "x"]}, ["channel", "space", "space", "space"]),
        (3, {"names": ["t", "y", "x"]}, ["time", "space", "space"]),
        # Out-of-order ZCYX (the regression): z->space and c->channel, not the
        # canonical-order positional default (which gave z->channel, c->space).
        (4, {"names": ["z", "c", "y", "x"]}, ["space", "channel", "space", "space"]),
        # Inference is case-insensitive.
        (4, {"names": ["Z", "C", "Y", "X"]}, ["space", "channel", "space", "space"]),
        # Explicit types are honored verbatim.
        (2, {"names": ["y", "x"], "types": ["custom", "space"]}, ["custom", "space"]),
        # Unknown/custom axis name -> untyped (None); NGFF permits this.
        (3, {"names": ["q", "y", "x"]}, [None, "space", "space"]),
    ],
)
def test_axes_types(
    ndim: int, kwargs: Dict[str, Any], expected_types: List[Optional[str]]
) -> None:
    ax = Axes(ndim=ndim, **kwargs)
    assert ax.types == expected_types
    meta = ax.to_metadata()
    assert [d.get("type") for d in meta] == expected_types
    assert all(("type" in d) == (t is not None) for d, t in zip(meta, expected_types))
