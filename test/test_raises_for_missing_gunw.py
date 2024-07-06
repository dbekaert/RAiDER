'''
Regression tests for issue #648:
Bad error message when GUNW file missing in S3 bucket

Program should raise an error if the GUNW product file, metadata file,
or browse image is missing that clearly explains what went wrong, as opposed to
a generic error message resulting from a side effect of the error.
'''
from contextlib import contextmanager
from typing import List

import pytest
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path
import RAiDER.aws
import RAiDER.cli.raider


EXAMPLE_GUNW_PATH = 'test/gunw_test_data/S1-GUNW-D-R-059-tops-20230320_20220418-180300-00179W_00051N-PP-c92e-v2_0_6.nc'
EXAMPLE_JSON_DATA_PATH = 'test/gunw_test_data/S1-GUNW-A-R-064-tops-20210723_20210711-015001-35393N_33512N-PP-6267-v2_0_4.json'


@pytest.fixture
def iargs() -> List[str]:
    return [
        '--bucket', 'dummy-bucket',
        '--input-bucket-prefix', 'dummy-input-prefix',
        '--weather-model', 'ERA5',
    ]

@contextmanager
def make_gunw_path():
    with TemporaryDirectory() as tempdir:
        shutil.copy(EXAMPLE_GUNW_PATH, tempdir)
        yield Path(tempdir) / Path(EXAMPLE_GUNW_PATH).name

@contextmanager
def make_json_data_path():
    with TemporaryDirectory() as tempdir:
        shutil.copy(EXAMPLE_JSON_DATA_PATH, tempdir)
        yield Path(tempdir) / Path(EXAMPLE_JSON_DATA_PATH).name


# Patch aws.get_s3_file to produce None then check for the correct error message
def test_missing_product_file(mocker, iargs):
    side_effect = [
        None,  # GUNW product file
        # program should fail
    ]
    mocker.patch('RAiDER.aws.get_s3_file', side_effect=side_effect)
    with pytest.raises(ValueError) as excinfo:
        RAiDER.cli.raider.calcDelaysGUNW(iargs)
    assert "GUNW product file could not be found" in str(excinfo.value)


# Patch aws.get_s3_file to produce None then check for the correct error message
def test_missing_metadata_file(mocker, iargs):
    with make_gunw_path() as gunw_path:
        side_effect = [
            gunw_path,  # GUNW product file
            None,  # GUNW metadata file
            # program should fail
        ]
        mocker.patch('RAiDER.aws.get_s3_file', side_effect=side_effect)
        with pytest.raises(ValueError) as excinfo:
            RAiDER.cli.raider.calcDelaysGUNW(iargs)
        assert "GUNW metadata file could not be found" in str(excinfo.value)


def test_missing_browse_image(mocker, iargs):
    with make_gunw_path() as gunw_path, make_json_data_path() as json_data_path:
        side_effect = [
            gunw_path,  # GUNW product file
            json_data_path,  # GUNW metadata file
            None,  # GUNW browse image
            # program should fail
        ]
        mocker.patch('RAiDER.aws.get_s3_file', side_effect=side_effect)
        with pytest.raises(ValueError) as excinfo:
            RAiDER.cli.raider.calcDelaysGUNW(iargs)
        assert "GUNW browse image could not be found" in str(excinfo.value)
