import datetime as dt
import logging
from pathlib import Path
from typing import Type

import numpy as np
import pytest
from requests.exceptions import HTTPError

from RAiDER.models import ERA5, ERA5T, GMAO, HRES, MERRA2
from RAiDER.models.erai import ERAI
from RAiDER.models.weatherModel import WeatherModel
from test import random_string


DATETIME = dt.datetime(2020, 1, 1, 0, 0, 0).replace(tzinfo=dt.timezone(offset=dt.timedelta()))
BOUNDS = np.array([10, 10.2, -72, -72])


@pytest.mark.long
@pytest.mark.parametrize(
    'name,Model',
    [
        ('ERA5', ERA5),
        ('ERA5T', ERA5T),
        pytest.param('HRES', HRES, marks=pytest.mark.skip),  # Paid access
        ('GMAO', GMAO),
        ('MERRA2', MERRA2),
    ],
)
def test_downloader(tmp_path: Path, name: str, Model: Type[WeatherModel]) -> None:
    out_path = tmp_path / f'test_{name}.nc'
    wm = Model()
    wm.set_latlon_bounds(BOUNDS)
    wm.fetch(out_path, DATETIME)


@pytest.mark.long
def test_erai(tmp_path: Path) -> None:
    out_path = tmp_path / 'test_erai.nc'
    wm = ERAI()
    wm.set_latlon_bounds(BOUNDS)
    wm.fetch(out_path, dt.datetime(2017, 1, 1, 0, 0, 0).replace(tzinfo=dt.timezone(offset=dt.timedelta())))


def test_old_api_url_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    out_path = tmp_path / 'test_era5-old-api-url.nc'
    wm = ERA5()
    wm.set_latlon_bounds(BOUNDS)
    with monkeypatch.context() as mp:
        mp.setenv('CDSAPI_URL', 'https://cds.climate.copernicus.eu/api/v2')
        mp.setenv('CDSAPI_KEY', random_string())
        with caplog.at_level(logging.WARNING), pytest.raises(HTTPError, match='404'):
            wm.fetch(out_path, DATETIME)
    assert 'Old CDS API configuration detected' in caplog.text
