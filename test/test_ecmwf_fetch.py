"""Tests for the ERA-5 model-level download post-processing.

`ECMWF._get_from_cds` writes the file that `ECMWF._makeDataCubes` later reads, so
the two have to agree on the layout of the surface fields. Nothing in CI performs
a real CDS retrieval, so these tests stub out `cdsapi.Client` and exercise the
post-processing and the reader against each other.

The layout contract is:

* `t`, `q`, `z` are ``(time, level, lat, lon)`` cubes spanning all model levels
* `lnsp` is a surface field, recovered by the reader as ``lnsp[0, 0]``
* `z_surface` is the downloaded surface geopotential, kept alongside the
  recomputed full-level `z` cube and recovered the same way as `lnsp`; the
  reader starts the hydrostatic integration from it, falling back to the
  lowest full level for files fetched before it was stored

ERA-5 archives `lnsp` and `z` at model level 1 only, so they have to be
requested separately from `t`/`q`: CDS returns a mixed-level request as separate
files rather than as a single netCDF.
"""

import datetime as dt
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from RAiDER.constants import _g0
from RAiDER.models.era5 import ERA5
from RAiDER.utilFcns import geo_to_ht


NLEV = 137
LATS = np.arange(50.0, 48.9, -0.25)  # descending, as ERA-5 delivers them
LONS = np.arange(9.0, 10.1, 0.25)
NLAT, NLON = len(LATS), len(LONS)

TIME = np.array(['2020-01-01T00:00:00'], dtype='datetime64[ns]')
SURFACE_PRESSURE = 95_000.0  # Pa
SURFACE_GEOPOTENTIAL = 500.0 * 9.80665  # 500 m, in m^2/s^2


def _t() -> np.ndarray:
    """Temperature decreasing from the surface (level 137) up to the top (level 1)."""
    profile = np.linspace(220.0, 290.0, NLEV)
    return np.broadcast_to(profile[:, np.newaxis, np.newaxis], (NLEV, NLAT, NLON)).copy()


def _q() -> np.ndarray:
    return np.full((NLEV, NLAT, NLON), 1e-3)


def _write_surface_file(path: Path) -> None:
    """Write the lnsp/z response: a single model level (level 1)."""
    lnsp = np.full((NLAT, NLON), np.log(SURFACE_PRESSURE))
    z = np.full((NLAT, NLON), SURFACE_GEOPOTENTIAL)
    xr.Dataset(
        {
            'lnsp': (('valid_time', 'model_level', 'latitude', 'longitude'), lnsp[np.newaxis, np.newaxis]),
            'z': (('valid_time', 'model_level', 'latitude', 'longitude'), z[np.newaxis, np.newaxis]),
        },
        coords={'valid_time': TIME, 'model_level': [1], 'latitude': LATS, 'longitude': LONS},
    ).to_netcdf(path)


def _write_model_level_file(path: Path) -> None:
    """Write the t/q response: all model levels."""
    xr.Dataset(
        {
            't': (('valid_time', 'model_level', 'latitude', 'longitude'), _t()[np.newaxis]),
            'q': (('valid_time', 'model_level', 'latitude', 'longitude'), _q()[np.newaxis]),
        },
        coords={
            'valid_time': TIME,
            'model_level': np.arange(1, NLEV + 1),
            'latitude': LATS,
            'longitude': LONS,
        },
    ).to_netcdf(path)


class MixedLevelRequestError(Exception):
    """Raised when a request mixes surface-only and model-level fields."""


class FakeCDSClient:
    """Stand-in for `cdsapi.Client` that writes synthetic ERA-5 responses.

    Records every request so tests can assert on how the download was split, and
    refuses to serve a request that mixes surface-only and model-level fields,
    which is what CDS does in practice.
    """

    url = 'https://cds.climate.copernicus.eu/api'
    requests: list[list[str]] = []

    def __init__(self, *args, **kwargs) -> None:
        FakeCDSClient.requests = []

    def retrieve(self, dataset: str, params: dict, target) -> None:
        param = sorted(params['param'])
        FakeCDSClient.requests.append(param)

        if param == ['lnsp', 'z']:
            _write_surface_file(Path(target))
        elif param == ['q', 't']:
            _write_model_level_file(Path(target))
        else:
            raise MixedLevelRequestError(
                f'CDS cannot serve {param} as a single netCDF: lnsp and z exist only at '
                'model level 1, so they must be requested separately from t and q.'
            )


@pytest.fixture()
def era5() -> ERA5:
    model = ERA5()
    model.setLevelType('ml')
    # Comfortably inside the synthetic grid, so the mask in _makeDataCubes keeps
    # every point and the shape assertions below are unambiguous.
    model.set_latlon_bounds([49.0, 50.0, 9.0, 10.0])
    return model


@pytest.fixture()
def fetched(era5: ERA5, tmp_path: Path) -> Path:
    out_path = tmp_path / 'ERA-5_2020_01_01_T00_00_00.nc'
    with patch('cdsapi.Client', FakeCDSClient):
        era5._get_from_cds(49.0, 50.0, 9.0, 10.0, dt.datetime(2020, 1, 1), out_path)
    return out_path


def test_surface_and_model_level_fields_are_requested_separately(fetched: Path) -> None:
    assert FakeCDSClient.requests == [['lnsp', 'z'], ['q', 't']]


def test_downloaded_file_is_readable_by_makeDataCubes(era5: ERA5, fetched: Path) -> None:
    """The shapes _load_model_level relies on when it calls _calculategeoh."""
    lats, lons, _, _, t, q, lnsp, z, z_surface = era5._makeDataCubes(fetched)

    assert lnsp.shape == (NLAT, NLON)
    assert t.shape == (NLEV, NLAT, NLON)
    assert q.shape == (NLEV, NLAT, NLON)
    assert z.shape == (NLEV, NLAT, NLON)
    assert z_surface.shape == (NLAT, NLON)
    assert len(lats) == NLAT
    assert len(lons) == NLON


def test_lnsp_round_trips_as_a_surface_field(era5: ERA5, fetched: Path) -> None:
    *_, lnsp, _, _ = era5._makeDataCubes(fetched)

    assert np.all(np.isfinite(lnsp))
    assert np.allclose(np.exp(lnsp), SURFACE_PRESSURE)


def test_z_is_written_as_a_full_model_level_cube(era5: ERA5, fetched: Path) -> None:
    """CDS supplies z at the surface only; _get_from_cds fills in the other levels."""
    *_, z, _ = era5._makeDataCubes(fetched)

    assert np.all(np.isfinite(z))
    # Level 137 is nearest the surface and level 1 is the top of the atmosphere,
    # so geopotential must increase monotonically towards index 0.
    assert np.all(np.diff(z, axis=0) < 0)
    assert np.allclose(z[-1], SURFACE_GEOPOTENTIAL, rtol=0.1)


def test_load_model_level_populates_the_model(era5: ERA5, fetched: Path) -> None:
    """End to end: the downloaded file drives a complete weather model."""
    era5._load_model_level(fetched)

    assert era5._p.shape == era5._t.shape == era5._q.shape == era5._zs.shape
    assert era5._t.shape[:2] == (NLAT, NLON)
    assert np.all(np.isfinite(era5._p))
    # _load_model_level flips everything to run surface-to-top.
    assert np.all(np.diff(era5._zs, axis=2) > 0)


def test_z_surface_round_trips_as_a_surface_field(era5: ERA5, fetched: Path) -> None:
    """The downloaded surface geopotential survives the post-processing unchanged."""
    *_, z_surface = era5._makeDataCubes(fetched)

    assert z_surface is not None
    assert np.array_equal(z_surface, np.full((NLAT, NLON), SURFACE_GEOPOTENTIAL))


def test_column_heights_start_at_the_surface(era5: ERA5, fetched: Path) -> None:
    """The loader reproduces the heights the fetch step computed.

    The loader re-runs the hydrostatic integration from the surface
    geopotential; the fetch step already did the same integration from the
    same inputs when it built the stored z cube, so the reconstructed lowest
    full level must match the stored one to numerical precision. Starting the
    integration from the lowest full level instead (the pre-fix behavior)
    shifts every column up by that level's height above the terrain (~10 m
    here), which this tolerance rejects.
    """
    *_, z, _ = era5._makeDataCubes(fetched)
    era5._load_model_level(fetched)

    # stored cube runs top -> surface with descending latitude; _zs runs
    # surface -> top with ascending latitude
    expected_lowest = geo_to_ht(era5._lats, z[-1, ::-1] / _g0)
    # The tolerance leaves room for a separate, much smaller defect: calcgeoh
    # turns its t argument into virtual temperature in place, and the fetch
    # step writes that mutated array back out, so the loader re-applies the
    # moisture factor and lands slightly above the stored cube — for this
    # fixture (q = 1e-3, T <= 290 K) roughly R_d * 0.609e-3 * 290 * alpha / g0
    # ~ 6 mm at the lowest level. The bug this test guards is three orders of
    # magnitude above the 0.02 m tolerance.
    assert np.allclose(era5._zs[..., 0], expected_lowest, atol=0.02)


def test_legacy_file_without_z_surface_falls_back(era5: ERA5, fetched: Path, tmp_path: Path) -> None:
    """Files fetched before z_surface was stored still load, with the old offset."""
    legacy_path = tmp_path / 'legacy.nc'
    with xr.open_dataset(fetched) as ds:
        ds.drop_vars('z_surface').to_netcdf(legacy_path)

    *_, z, z_surface = era5._makeDataCubes(legacy_path)
    assert z_surface is None

    era5._load_model_level(legacy_path)
    zs_legacy = era5._zs.copy()
    era5._load_model_level(fetched)

    # The fallback starts the integration at the lowest full level, so every
    # column sits higher than the fixed one by exactly that level's height
    # above the terrain.
    lowest_level_agl = (z[-1, ::-1] - SURFACE_GEOPOTENTIAL) / _g0
    assert np.allclose(zs_legacy[..., 0] - era5._zs[..., 0], lowest_level_agl, atol=1e-2)
