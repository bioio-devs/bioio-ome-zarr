import os
from typing import Generator
from uuid import uuid4

import fsspec
import numpy as np
import pytest

from bioio_ome_zarr.tests.writers.test_ome_zarr_writer import assert_valid_ome_zarr
from bioio_ome_zarr.writers import Channel, OMEZarrWriter

# aics-pipeline-output is located in the aics-dev account
S3_TEST_BUCKET = os.getenv("S3_TEST_BUCKET", "aics-pipeline-output")


@pytest.mark.skipif(
    os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Integration tests are disabled",
)
class TestS3WriterIntegration:
    """
    Validates that the OME-Zarr writer can write to S3. This test requires
    AWS credentials to be set in the environment and the ENABLE_INTEGRATION_TESTS
    environment variable to be set to "true".
    """

    @pytest.fixture(scope="class")
    @classmethod
    def s3_prefix(cls) -> Generator[str, None, None]:
        # Setup: Runs once before any tests in the class start
        s3_prefix = f"s3://{S3_TEST_BUCKET}/{uuid4()}"

        yield s3_prefix  # Provide the resource to tests

        # Teardown: Runs once after all tests in the class finish
        fsspec.filesystem("s3").rm(s3_prefix, recursive=True)

    def test_fsspec_writes_to_s3(self, s3_prefix: str) -> None:
        """
        Sanity check that fsspec can write to S3.
        """
        # Arrange
        s3_path = f"{s3_prefix}/test_fsspec_write.txt"
        data_to_write = b"Hello, S3!"

        # Act
        # Write the data to S3
        with fsspec.open(s3_path, "wb") as f:
            f.write(data_to_write)

        # Read the data back from S3
        with fsspec.open(s3_path, "rb") as f:
            read_data = f.read()

        # Assert
        assert read_data == data_to_write

    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_ome_zarr_writer_writes_to_s3(
        self, s3_prefix: str, zarr_format: int
    ) -> None:
        """
        Test that the OME-Zarr writer can write to S3. Simple example data taken
        from the README.
        """
        # Arrange
        level_shapes = [
            (2, 3, 4, 256, 256),  # L0 full res
            (2, 3, 4, 128, 128),  # L1 downsampled Y/X by 2
        ]

        data = np.random.randint(0, 255, size=level_shapes[0], dtype=np.uint8)
        channels = [
            Channel(label=f"c{i}", color="FF0000") for i in range(data.shape[1])
        ]

        output_store = f"{s3_prefix}/output_{zarr_format}.zarr"

        kwargs = dict(
            store=output_store,
            level_shapes=level_shapes,
            dtype=data.dtype,
            zarr_format=zarr_format,
            channels=channels,
            axes_names=["t", "c", "z", "y", "x"],
            axes_types=["time", "channel", "space", "space", "space"],
            axes_units=[None, None, "micrometer", "micrometer", "micrometer"],
        )
        writer = OMEZarrWriter(**kwargs)

        # Act
        writer.write_full_volume(data)

        # Assert
        # Check that the output store exists and is a directory
        fs = fsspec.filesystem("s3")
        assert fs.exists(output_store)
        assert fs.isdir(output_store)

        # Validate the OME-Zarr
        assert_valid_ome_zarr(output_store, zarr_format=zarr_format)
