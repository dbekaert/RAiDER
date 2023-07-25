from pathlib import Path
from typing import Callable

import pytest

test_dir = Path(__file__).parents[0]
TEST_DIR = test_dir.resolve()


def pytest_addoption(parser):
    parser.addoption(
        "--skip-isce3", action="store_true", default=False, help="skip tests which require ISCE3"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "isce3: mark test as requiring ISCE3 to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-isce3"):
        skip_isce3 = pytest.mark.skip(reason="--skip-isce3 option given")
        for item in items:
            if "isce3" in item.keywords:
                item.add_marker(skip_isce3)


@pytest.fixture(scope='session')
def test_dir_path() -> Path:
    return TEST_DIR


@pytest.fixture(scope='session')
def test_gunw_path_factory() -> Callable:
    def factory(location: str = 'california-t71') -> Path:
        if location == 'california-t71':
            file_name = 'S1-GUNW-D-R-071-tops-20200130_20200124-135156-34956N_32979N-PP-913f-v2_0_4.nc'
        elif location == 'alaska':
            file_name = 'S1-GUNW-D-R-059-tops-20230320_20220418-180300-00179W_00051N-PP-c92e-v2_0_6.nc'
        else:
            raise NotImplementedError
        return TEST_DIR / 'gunw_test_data' / file_name
    return factory


@pytest.fixture(scope='session')
def test_gunw_json_path() -> Path:
    p = TEST_DIR / 'gunw_test_data' / 'S1-GUNW-A-R-064-tops-20210723_20210711-015001-35393N_33512N-PP-6267-v2_0_4.json'
    return p


@pytest.fixture(scope='session')
def test_gunw_json_schema_path() -> Path:
    return TEST_DIR / 'gunw_test_data' / 'gunw_schema.json'
