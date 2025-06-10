import datetime as dt
import logging

import numpy as np
import pytest
from requests.exceptions import HTTPError

from RAiDER.models.era5 import ERA5
from RAiDER.models.era5t import ERA5T
from RAiDER.models.erai import ERAI
from test import TEST_DIR, random_string


@pytest.mark.long
def test_era5() -> None:
    wm = ERA5()
    wm.set_latlon_bounds(np.array([10, 10.2, -72, -72]))
    wm.fetch(
        TEST_DIR / 'test_geom/test_era5.nc',
        dt.datetime(2020, 1, 1, 0, 0, 0).replace(tzinfo=dt.timezone(offset=dt.timedelta())),
    )


@pytest.mark.long
def test_era5t() -> None:
    wm = ERA5T()
    wm.set_latlon_bounds(np.array([10, 10.2, -72, -72]))
    wm.fetch(
        TEST_DIR / 'test_geom/test_era5t.nc',
        dt.datetime(2020, 1, 1, 0, 0, 0).replace(tzinfo=dt.timezone(offset=dt.timedelta())),
    )


@pytest.mark.long
def test_erai() -> None:
    wm = ERAI()
    wm.set_latlon_bounds(np.array([10, 10.2, -72, -72]))
    wm.fetch(
        TEST_DIR / 'test_geom/test_erai.nc',
        dt.datetime(2017, 1, 1, 0, 0, 0).replace(tzinfo=dt.timezone(offset=dt.timedelta())),
    )


def test_old_api_url_warning(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as mp:
        mp.setenv('CDSAPI_URL', 'https://cds.climate.copernicus.eu/api/v2')
        mp.setenv('CDSAPI_KEY', random_string())
        wm = ERA5()
        wm.set_latlon_bounds(np.array([10, 10.2, -72, -72]))
        with caplog.at_level(logging.WARNING), pytest.raises(HTTPError, match="404"):
            wm.fetch(
                TEST_DIR / 'test_geom/test_era5.nc',
                dt.datetime(2020, 1, 1, 0, 0, 0).replace(tzinfo=dt.timezone(offset=dt.timedelta())),
            )
    assert 'Old CDS API configuration detected' in caplog.text
