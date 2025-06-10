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
        elif location == 'philippines':
            file_name = 'S1-GUNW-D-R-032-tops-20200220_20200214-214625-00120E_00014N-PP-b785-v3_0_1.nc'
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


@pytest.fixture(scope='session')
def gunw_azimuth_test():
    test_data = TEST_DIR / 'gunw_azimuth_test_data'
    return test_data / 'S1-GUNW-A-R-064-tops-20210723_20210711-015000-00119W_00033N-PP-6267-v2_0_6.nc'


@pytest.fixture(scope='session')
def orbit_dict_for_azimuth_time_test():
    test_data = TEST_DIR / 'gunw_azimuth_test_data'
    return {'reference': test_data / 'S1B_OPER_AUX_POEORB_OPOD_20210812T111941_V20210722T225942_20210724T005942.EOF',
            'secondary': test_data / 'S1B_OPER_AUX_POEORB_OPOD_20210731T111940_V20210710T225942_20210712T005942.EOF'}


@pytest.fixture(scope='session')
def slc_id_dict_for_azimuth_time_test():
    test_data = TEST_DIR / 'gunw_azimuth_test_data'
    return {'reference': [test_data / 'S1B_IW_SLC__1SDV_20210723T014947_20210723T015014_027915_0354B4_B3A9'],
            'secondary': [test_data / 'S1B_IW_SLC__1SDV_20210711T014922_20210711T014949_027740_034F80_859D',
                          test_data / 'S1B_IW_SLC__1SDV_20210711T015011_20210711T015038_027740_034F80_376C']}


@pytest.fixture(scope='session')
def weather_model_dict_for_azimuth_time_test():
    """The order is important; will be closest to InSAR acq time so goes 2, 1, 3 AM."""
    test_data = TEST_DIR / 'gunw_azimuth_test_data' / 'weather_files'
    return {'HRRR': [test_data / 'HRRR_2021_07_23_T02_00_00_33N_36N_120W_115W.nc',
                     test_data / 'HRRR_2021_07_23_T01_00_00_33N_36N_120W_115W.nc',
                     test_data / 'HRRR_2021_07_11_T02_00_00_33N_36N_120W_115W.nc',
                     test_data / 'HRRR_2021_07_11_T01_00_00_33N_36N_120W_115W.nc',
                     ]}


@pytest.fixture(scope='session')
def weather_model_dict_for_center_time_test():
    """Order is important here; will be in chronological order with respect to closest date times"""
    test_data = TEST_DIR / 'gunw_azimuth_test_data' / 'weather_files'
    return {'HRRR': [test_data / 'HRRR_2021_07_23_T01_00_00_33N_36N_120W_115W.nc',
                     test_data / 'HRRR_2021_07_23_T02_00_00_33N_36N_120W_115W.nc',
                     test_data / 'HRRR_2021_07_11_T01_00_00_33N_36N_120W_115W.nc',
                     test_data / 'HRRR_2021_07_11_T02_00_00_33N_36N_120W_115W.nc',
                     ]
            }


@pytest.fixture(scope='session')
def orbit_paths_for_duplicate_orbit_xml_test():
    test_data = TEST_DIR / 'data_for_overlapping_orbits'
    orbit_file_names = ['S1A_OPER_AUX_POEORB_OPOD_20230413T080643_V20230323T225942_20230325T005942.EOF',
                        'S1A_OPER_AUX_POEORB_OPOD_20230413T080643_V20230323T225942_20230325T005942.EOF',
                        'S1A_OPER_AUX_POEORB_OPOD_20230413T080643_V20230323T225942_20230325T005942.EOF',
                        'S1A_OPER_AUX_POEORB_OPOD_20230412T080821_V20230322T225942_20230324T005942.EOF']
    return [test_data / fn for fn in orbit_file_names]


@pytest.fixture(scope='session')
def weather_model_dict_for_gunw_integration_test():
    """Order is important here; will be in chronological order with respect to closest date times.

    Generate via:
    ```
    from RAiDER.processWM import prepareWeatherModel
    from RAiDER.models import GMAO
    import datetime

    model = GMAO()
    datetimes = [datetime.datetime(2020, 1, 30, 12, 0),
                 datetime.datetime(2020, 1, 30, 15, 0),
                 datetime.datetime(2020, 1, 24, 12, 0),
                 datetime.datetime(2020, 1, 24, 15, 0)]
    bounds = [32.5, 35.5, -119.8, -115.7]
    wmfiles = [prepareWeatherModel(model, dt, bounds) for dt in datetimes]
    ```
    """
    test_data = TEST_DIR / 'gunw_test_data' / 'weather_files'
    return {'GMAO': [test_data / 'GMAO_2020_01_30_T12_00_00_32N_36N_121W_114W.nc',
                     test_data / 'GMAO_2020_01_30_T15_00_00_32N_36N_121W_114W.nc',
                     test_data / 'GMAO_2020_01_24_T12_00_00_32N_36N_121W_114W.nc',
                     test_data / 'GMAO_2020_01_24_T15_00_00_32N_36N_121W_114W.nc']
           }

@pytest.fixture(scope='session')
def data_for_hrrr_ztd():
    '''Obtained via:
    ```
    from RAiDER.processWM import prepareWeatherModel
    from RAiDER.models import HRRR
    import datetime

    model = HRRR()
    datetimes = [datetime.datetime(2020, 1, 1, 12)]
    bounds = [36, 37, -92, -91]
    wmfiles = [prepareWeatherModel(model, dt, bounds) for dt in datetimes]
    ```
    '''
    test_data_dir = TEST_DIR / 'scenario_1' / 'HRRR_ztd_test'
    return test_data_dir / 'HRRR_2020_01_01_T12_00_00_35N_38N_93W_90W.nc'