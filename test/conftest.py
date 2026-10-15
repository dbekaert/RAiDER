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


# Weather-model fixture directories that are tracked in git. `combine_weather_files`
# writes its time-interpolated product next to its inputs, so a fixture that hands a
# test one of these paths instead of the scratch directory built by
# `_linked_weather_files` will overwrite tracked inputs or leave derived models in the repo.
# Nothing should ever appear here during a test run; the session hooks below turn that
# into a visible failure rather than something you find later in `git status`.
_TRACKED_WEATHER_DIRS = (
    TEST_DIR / 'weather_files',
    TEST_DIR / 'gunw_test_data' / 'weather_files',
    TEST_DIR / 'gunw_azimuth_test_data' / 'weather_files',
)


def _weather_fixture_files() -> set[Path]:
    """Every file currently sitting in a tracked weather-model fixture directory."""
    return {f for d in _TRACKED_WEATHER_DIRS for f in d.rglob('*') if f.is_file()}


def pytest_sessionstart(session):
    session.config._weather_fixture_files = _weather_fixture_files()


def pytest_sessionfinish(session, exitstatus):
    """Fail the run if a test deposited a weather model in a tracked fixture directory.

    Compares against the session-start snapshot rather than asserting the directories
    are empty, so pre-existing leftovers from an older checkout do not masquerade as a
    regression introduced by this run.
    """
    before = getattr(session.config, '_weather_fixture_files', None)
    if before is None:
        return
    written = sorted(str(f.relative_to(TEST_DIR)) for f in _weather_fixture_files() - before)
    if not written:
        return

    # Only promote a clean run to a failure. An interrupted or misconfigured
    # session already carries a more specific code, and overwriting it with
    # TESTS_FAILED would hide why the run actually stopped.
    if exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    if reporter is None:
        return
    reporter.write_sep('=', 'weather models written into tracked fixture directories', red=True)
    for name in written:
        reporter.write_line(f'  test/{name}')
    reporter.write_line(
        'A fixture handed a test a repo path instead of a scratch directory. '
        'Route it through _linked_weather_files in test/conftest.py, then delete the files above.'
    )


# One scratch directory per source directory, shared by every fixture that draws
# from it. Two fixtures expose the same gunw_azimuth_test_data/weather_files with
# different orderings; without this they would each get their own mktemp, so the
# two interpolation styles would write into separate directories instead of the
# single one they shared before the fixtures were redirected out of the repo.
_LINK_DIRS: dict[Path, Path] = {}


def _linked_weather_files(
    tmp_path_factory: pytest.TempPathFactory, src_dir: Path, names: list[str]
) -> list[Path]:
    """Expose weather-model files from a scratch directory instead of the repo.

    `combine_weather_files` writes the time-interpolated product next to its
    inputs (`wfiles[0].parent`, see RAiDER.cli.raider), so handing tests paths
    inside the tracked fixture directories makes every run deposit derived
    `_timeInterp_` / `_timeInterpAziGrid_` files there -- overwriting the
    checked-in copies under test/gunw_test_data, and rewriting ~165 MB into
    test/gunw_azimuth_test_data. The derived names come from the GUNW scene's
    fixed center time, so each run overwrites the last rather than piling up.

    Symlinking keeps that behaviour intact while redirecting the output: the
    links are read through transparently, nothing is copied, and
    `wfiles[0].parent` resolves to the scratch directory. Paths are not
    resolved anywhere in the delay workflow, so the symlink parent is what the
    writer sees.

    The redirect covers newly created files only. Symlinks are read-through for
    writes as well, so anything that rewrote one of these inputs in place would
    write straight into the tracked copy rather than into the scratch directory.
    Note also that `symlink_to` needs Developer Mode or an elevated shell on
    Windows; the suite is exercised on Linux, natively or under WSL.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.
        src_dir: Tracked directory holding the real weather-model files.
        names: File names to expose, in the order the caller needs them.

    Returns:
        Paths to the symlinks, in the same order as `names`.
    """
    work = _LINK_DIRS.get(src_dir)
    if work is None:
        work = _LINK_DIRS[src_dir] = tmp_path_factory.mktemp(src_dir.name)
    for name in names:
        src = src_dir / name
        # symlink_to happily creates a dangling link, which would otherwise
        # surface much later as a FileNotFoundError on a /tmp path that says
        # nothing about which fixture name was wrong.
        if not src.exists():
            raise FileNotFoundError(f'missing weather-model fixture: {src}')
        link = work / name
        if not link.exists():
            link.symlink_to(src)
    return [work / name for name in names]


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
def weather_model_dict_for_azimuth_time_test(tmp_path_factory):
    """The order is important; will be closest to InSAR acq time so goes 2, 1, 3 AM."""
    test_data = TEST_DIR / 'gunw_azimuth_test_data' / 'weather_files'
    return {'HRRR': _linked_weather_files(tmp_path_factory, test_data, [
        'HRRR_2021_07_23_T02_00_00_33N_36N_120W_115W.nc',
        'HRRR_2021_07_23_T01_00_00_33N_36N_120W_115W.nc',
        'HRRR_2021_07_11_T02_00_00_33N_36N_120W_115W.nc',
        'HRRR_2021_07_11_T01_00_00_33N_36N_120W_115W.nc',
    ])}


@pytest.fixture(scope='session')
def weather_model_dict_for_center_time_test(tmp_path_factory):
    """Order is important here; will be in chronological order with respect to closest date times"""
    test_data = TEST_DIR / 'gunw_azimuth_test_data' / 'weather_files'
    return {'HRRR': _linked_weather_files(tmp_path_factory, test_data, [
        'HRRR_2021_07_23_T01_00_00_33N_36N_120W_115W.nc',
        'HRRR_2021_07_23_T02_00_00_33N_36N_120W_115W.nc',
        'HRRR_2021_07_11_T01_00_00_33N_36N_120W_115W.nc',
        'HRRR_2021_07_11_T02_00_00_33N_36N_120W_115W.nc',
    ])}


@pytest.fixture(scope='session')
def weather_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Scratch stand-in for the tracked test/weather_files directory.

    Tests that set `weather_model_directory` are handed a directory of symlinks
    rather than the tracked one. `prepareWeatherModel` only writes a model when
    the file it wants is absent, and `os.path.exists` follows symlinks, so the
    committed crops are still found and nothing is downloaded. The difference is
    where a *miss* lands: a bounding box that no committed crop covers now writes
    the fresh model into the scratch directory instead of into the repository.

    Note that symlinks are read-through for writes as well, so this redirects
    newly created files only. Anything that rewrote a weather file in place would
    still write through into the tracked copy.

    Returns:
        The scratch directory to pass as `weather_model_directory`.
    """
    src_dir = TEST_DIR / 'weather_files'
    names = sorted(f.name for f in src_dir.iterdir() if f.is_file())
    return _linked_weather_files(tmp_path_factory, src_dir, names)[0].parent


@pytest.fixture(scope='session')
def orbit_paths_for_duplicate_orbit_xml_test():
    test_data = TEST_DIR / 'data_for_overlapping_orbits'
    orbit_file_names = ['S1A_OPER_AUX_POEORB_OPOD_20230413T080643_V20230323T225942_20230325T005942.EOF',
                        'S1A_OPER_AUX_POEORB_OPOD_20230413T080643_V20230323T225942_20230325T005942.EOF',
                        'S1A_OPER_AUX_POEORB_OPOD_20230413T080643_V20230323T225942_20230325T005942.EOF',
                        'S1A_OPER_AUX_POEORB_OPOD_20230412T080821_V20230322T225942_20230324T005942.EOF']
    return [test_data / fn for fn in orbit_file_names]


@pytest.fixture(scope='session')
def weather_model_dict_for_gunw_integration_test(tmp_path_factory):
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
    return {'GMAO': _linked_weather_files(tmp_path_factory, test_data, [
        'GMAO_2020_01_30_T12_00_00_32N_36N_121W_114W.nc',
        'GMAO_2020_01_30_T15_00_00_32N_36N_121W_114W.nc',
        'GMAO_2020_01_24_T12_00_00_32N_36N_121W_114W.nc',
        'GMAO_2020_01_24_T15_00_00_32N_36N_121W_114W.nc',
    ])}


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
