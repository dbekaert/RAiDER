from pathlib import Path

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
def test_gunw_path() -> Path:
    return TEST_DIR / 'gunw_test_data' / 'S1-GUNW-D-R-071-tops-20200130_20200124-135156-34956N_32979N-PP-913f-v2_0_4.nc'


@pytest.fixture(scope='session')
def test_gunw_json_path() -> Path:
    p = TEST_DIR / 'gunw_test_data' / 'S1-GUNW-A-R-064-tops-20210723_20210711-015001-35393N_33512N-PP-6267-v2_0_4.json'
    return p


@pytest.fixture(scope='session')
def test_gunw_json_schema_path() -> Path:
    return TEST_DIR / 'gunw_test_data' / 'gunw_schema.json'
