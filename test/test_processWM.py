import datetime as dt
import os
from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from pytest_mock import MockerFixture

from RAiDER import processWM
from RAiDER.models.customExceptions import (
    CriticalError,
    DatetimeOutsideRange,
    ExistingWeatherModelTooSmall,
    TryToKeepGoingError,
)
from RAiDER.models.era5 import ERA5
from RAiDER.models.weatherModel import WeatherModel
from test import TEST_DIR


SCENARIO_DIR = Path(TEST_DIR) / 'scenario_1'


@pytest.fixture
def getWM():  # noqa: ANN201
    """
    The bounds of the ERA model are:
    BoundingBox(left=-103.7750015258789,
                bottom=15.225000381469727,
                right=-99.2750015258789,
                top=18.725000381469727).
    """
    wm = ERA5()
    wm.files = [Path(SCENARIO_DIR) / 'weather_files' / 'ERA-5_2020_01_03_T23_00_00.nc']
    return wm


def test_checkContainment(getWM) -> None:  # noqa: ANN001
    """Test whether a weather model contains a bbox."""
    wm = getWM
    if Path.exists(wm.files[0]):
        ll_bounds = (10, 20, -100, -100)
        containment = wm.checkContainment(ll_bounds)
        assert ~containment


def test_checkContainment2(getWM) -> None:  # noqa: ANN001
    """Test whether a weather model contains a bbox."""
    wm = getWM
    if Path.exists(wm.files[0]):
        ll_bounds = (17, 18, -100, -100)
        containment = wm.checkContainment(ll_bounds)
        assert containment


def test_checkContainment3() -> None:  # noqa: ANN001
    """Test whether a weather model contains a bbox."""
    with pytest.raises(TypeError):
        WeatherModel()


def test_checkContainment4() -> None:  # noqa: ANN001
    """Test whether a weather model contains a bbox."""
    wm = ERA5()
    wm._bbox = [-180, 0, 180, 90]
    ll_bounds = (1, 89, -179, 179)
    assert wm.checkContainment(ll_bounds)


def test_checkContainment5() -> None:  # noqa: ANN001
    """Test whether a weather model contains a bbox."""
    wm = ERA5()
    wm._bbox = [-180, 0, 180, 90]
    ll_bounds = (0, 90, -180, 180)
    assert wm.checkContainment(ll_bounds)


TIME = dt.datetime(2020, 1, 3, 23, 0, 0)
LL_BOUNDS = (17, 18, -101, -100)


@pytest.fixture
def raw_path(tmp_path: Path, mocker: MockerFixture) -> Path:
    """Point the raw weather model file at a temporary directory that does not exist yet."""
    path = tmp_path / 'raw' / 'ERA5_raw.nc'
    mocker.patch.object(processWM, 'make_raw_weather_data_filename', return_value=str(path))
    return path


@pytest.fixture
def fake_wm(tmp_path: Path, raw_path: Path) -> MagicMock:
    """A stand-in weather model whose files live in a temporary directory and are not yet processed."""
    wm = MagicMock()
    wm.get_latlon_bounds.return_value = LL_BOUNDS
    wm.get_wmLoc.return_value = tmp_path
    wm.Model.return_value = 'ERA5'
    wm.out_file.return_value = tmp_path / 'ERA5_cropped.nc'
    wm.load.return_value = None
    wm.getWetRefractivity.return_value = np.ones((2, 3, 4))
    wm.getHydroRefractivity.return_value = np.ones((2, 3, 4))
    wm._ys = np.array([17.0, 18.0])
    wm._xs = np.array([-101.0, -100.0])
    wm.write.return_value = 'ERA5_written.nc'
    wm.checkContainment.return_value = True
    return wm


def test_prepareWeatherModel_sets_missing_bounds(fake_wm: MagicMock) -> None:
    """The requested bounds are used when the weather model has none."""
    fake_wm.get_latlon_bounds.return_value = None
    processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS)
    fake_wm.set_latlon_bounds.assert_called_once_with(LL_BOUNDS)
    fake_wm.setTime.assert_called_once_with(TIME)


def test_prepareWeatherModel_keeps_existing_bounds(fake_wm: MagicMock) -> None:
    """Bounds already set on the weather model are not overwritten."""
    processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS)
    fake_wm.set_latlon_bounds.assert_not_called()


def test_prepareWeatherModel_downloads_when_no_files(fake_wm: MagicMock, raw_path: Path) -> None:
    """With no raw or processed file, the raw directory is created and the data is fetched."""
    processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS)
    assert raw_path.parent.is_dir()
    fake_wm.fetch.assert_called_once_with(raw_path, TIME)


def test_prepareWeatherModel_skips_download_when_processed_exists(fake_wm: MagicMock) -> None:
    """An existing processed file is not downloaded again."""
    Path(fake_wm.out_file.return_value).touch()
    processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS)
    fake_wm.fetch.assert_not_called()


def test_prepareWeatherModel_force_download(fake_wm: MagicMock, raw_path: Path, mocker: MockerFixture) -> None:
    """force_download fetches even when the processed and raw files exist."""
    Path(fake_wm.out_file.return_value).touch()
    raw_path.parent.mkdir()
    raw_path.touch()
    mocker.patch.object(processWM, 'checkContainment_raw', return_value=True)
    processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS, force_download=True)
    fake_wm.fetch.assert_called_once_with(raw_path, TIME)


@pytest.mark.parametrize('contains, fetched', [(True, False), (False, True)])
def test_prepareWeatherModel_existing_raw(
    fake_wm: MagicMock, raw_path: Path, mocker: MockerFixture, contains: bool, fetched: bool
) -> None:
    """An existing raw file is reused only if it covers the requested bounds."""
    raw_path.parent.mkdir()
    raw_path.touch()
    check = mocker.patch.object(processWM, 'checkContainment_raw', return_value=contains)
    processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS)
    check.assert_called_once_with(str(raw_path), LL_BOUNDS)
    assert fake_wm.fetch.called is fetched


def test_prepareWeatherModel_datetime_outside_range(fake_wm: MagicMock) -> None:
    """A time the model does not cover is turned into a TryToKeepGoingError."""
    fake_wm.fetch.side_effect = DatetimeOutsideRange('ERA5', TIME)
    with pytest.raises(TryToKeepGoingError):
        processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS)


def test_prepareWeatherModel_download_only(fake_wm: MagicMock) -> None:
    """download_only stops after fetching and returns None."""
    assert processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS, download_only=True) is None
    fake_wm.fetch.assert_called_once()
    fake_wm.load.assert_not_called()


def test_prepareWeatherModel_uses_loaded_file(fake_wm: MagicMock) -> None:
    """A processed file returned by load() is returned without writing a new one."""
    fake_wm.load.return_value = 'ERA5_loaded.nc'
    assert processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS) == 'ERA5_loaded.nc'
    fake_wm.checkContainment.assert_called_once_with(LL_BOUNDS)
    fake_wm.write.assert_not_called()


def test_prepareWeatherModel_loaded_file_too_small(fake_wm: MagicMock) -> None:
    """A loaded file that does not cover the bounds is rejected."""
    fake_wm.load.return_value = 'ERA5_loaded.nc'
    fake_wm.checkContainment.return_value = False
    with pytest.raises(ExistingWeatherModelTooSmall):
        processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS)


@pytest.mark.parametrize('model', ['HRRR', 'HRRRAK'])
def test_prepareWeatherModel_loaded_file_too_small_hrrr(fake_wm: MagicMock, model: str) -> None:
    """HRRR models are allowed to not fully cover the bounds."""
    fake_wm.Model.return_value = model
    fake_wm.load.return_value = 'HRRR_loaded.nc'
    fake_wm.checkContainment.return_value = False
    assert processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS) == 'HRRR_loaded.nc'


def test_prepareWeatherModel_writes_processed_file(fake_wm: MagicMock) -> None:
    """A newly processed weather model is written and its file name returned."""
    assert processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS) == 'ERA5_written.nc'
    fake_wm.write.assert_called_once_with()
    fake_wm.checkContainment.assert_called_once_with(LL_BOUNDS)
    fake_wm.plot.assert_not_called()


def test_prepareWeatherModel_makes_plots(fake_wm: MagicMock, mocker: MockerFixture) -> None:
    """makePlots writes both debug plots and closes the figures."""
    close = mocker.patch.object(processWM.plt, 'close')
    processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS, makePlots=True)
    assert fake_wm.plot.mock_calls == [call('wh', True), call('pqt', True)]
    close.assert_called_once_with('all')


def test_prepareWeatherModel_written_file_too_small(fake_wm: MagicMock) -> None:
    """A written file that does not cover the bounds is rejected."""
    fake_wm.checkContainment.return_value = False
    with pytest.raises(ExistingWeatherModelTooSmall):
        processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS)


def test_prepareWeatherModel_written_file_too_small_hrrr(fake_wm: MagicMock) -> None:
    """HRRR is allowed to not fully cover the bounds after writing."""
    fake_wm.Model.return_value = 'HRRR'
    fake_wm.checkContainment.return_value = False
    assert processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS) == 'ERA5_written.nc'


def test_prepareWeatherModel_write_fails(fake_wm: MagicMock) -> None:
    """A failure while writing is raised as a CriticalError chained to the original error."""
    fake_wm.write.side_effect = OSError('disk full')
    with pytest.raises(CriticalError) as excinfo:
        processWM.prepareWeatherModel(fake_wm, TIME, LL_BOUNDS)
    assert isinstance(excinfo.value.__cause__, OSError)
