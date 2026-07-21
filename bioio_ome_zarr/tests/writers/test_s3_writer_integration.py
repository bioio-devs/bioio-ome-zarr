import os
from typing import Generator
from uuid import uuid4

import fsspec
import pytest

# located in the aics-dev account (771753870375)
S3_TEST_BUCKET_NAME = "aics-pipeline-output"


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
    def s3_prefix(cls) -> Generator[str]:
        # Setup: Runs once before any tests in the class start
        s3_prefix = f"s3://{S3_TEST_BUCKET_NAME}/{uuid4()}"

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
