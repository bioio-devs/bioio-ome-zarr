import pathlib
import typing

import dask.array as da
import numpy as np
import pytest

LOCAL_RESOURCES_DIR = pathlib.Path(__file__).parent / "resources"


def np_random_from_shape(
    shape: typing.Tuple[int, ...], **kwargs: typing.Any
) -> np.ndarray:
    return np.random.randint(255, size=shape, **kwargs)


def da_random_from_shape(
    shape: typing.Tuple[int, ...], **kwargs: typing.Any
) -> da.Array:
    return da.random.randint(255, size=shape, **kwargs)


array_constructor = pytest.mark.parametrize(
    "array_constructor",
    [np_random_from_shape, da_random_from_shape],
)
