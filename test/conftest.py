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


@pytest.fixture(scope='session')
def gunw_azimuth_test():
    test_data = TEST_DIR / 'gunw_azimuth_test_data'
    return test_data / 'S1-GUNW-A-R-064-tops-20210723_20210711-015000-00119W_00033N-PP-6267-v2_0_6.nc'


@pytest.fixture(scope='session')
def orbit_dict_for_azimuth_test():
    test_data = TEST_DIR / 'gunw_azimuth_test_data'
    return {'reference': test_data / 'S1B_OPER_AUX_POEORB_OPOD_20210812T111941_V20210722T225942_20210724T005942.EOF',
            'secondary': test_data / 'S1B_OPER_AUX_POEORB_OPOD_20210731T111940_V20210710T225942_20210712T005942.EOF'}


@pytest.fixture(scope='session')
def slc_id_dict_for_azimuth_test():
    test_data = TEST_DIR / 'gunw_azimuth_test_data'
    return {'reference': test_data / 'S1B_IW_SLC__1SDV_20210723T014947_20210723T015014_027915_0354B4_B3A9',
            'secondary': test_data / 'S1B_IW_SLC__1SDV_20210711T014947_20210711T015013_027740_034F80_D404'}
