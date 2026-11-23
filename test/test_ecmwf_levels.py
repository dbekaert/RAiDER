"""Regression tests for the ECMWF/ERA-5 model-level reader.

These cover the orientation of the geopotential cube relative to the other
fields. ECMWF delivers model-level data with latitude descending, and the
reader reverses it to ascending; the height cube has to be reversed on the
same axis as t/q/lnsp, or heights end up mirrored north-south against the
meteorology.

That failure mode is antisymmetric in latitude, so it very nearly cancels in
a domain average -- a mean-based check will not catch it. The assertions here
are deliberately per-column.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pytest
import xarray as xr

from RAiDER.constants import _g0
from RAiDER.models.era5 import ERA5
from RAiDER.utilFcns import calcgeoh, geo_to_ht


# Terrain rising steeply from south to north, so that a latitude reflection is
# unambiguous: the southern row is at sea level, the northern row is high. It
# also rises gently from west to east, so a longitude reflection (or a missed
# one) shows up too.
_LATS_DESC = np.array([36.0, 35.0, 34.0, 33.0])  # ECMWF order: descending
_LONS_ASC = np.array([240.0, 241.0, 242.0])  # 0-360, as ECMWF delivers
_LAT_SLOPE = 1000.0  # m per degree north
_LON_SLOPE = 100.0  # m per degree east


def _terrain(lat: Union[float, np.ndarray], lon: Union[float, np.ndarray]) -> np.ndarray:
    """Synthetic terrain height (m) at (lat, lon); lon may be in either 0-360 or -180..180."""
    return (np.asarray(lat) - 33.0) * _LAT_SLOPE + ((np.asarray(lon) % 360) - 240.0) * _LON_SLOPE


def _write_synthetic_ml_file(path: Path, include_z_surface: bool = False, lons_descending: bool = False) -> ERA5:
    """Write a minimal ERA-5 model-level file in ECMWF's native ordering.

    Mirrors what RAiDER's own _fetch writes: z is the *full* geopotential cube
    (computed with calcgeoh), not the surface-only field CDS returns. With
    ``include_z_surface`` the downloaded surface geopotential is kept as its
    own variable, as the fetch step now does; without it the file has the
    legacy layout, where the reader has to fall back to the lowest full level.
    With ``lons_descending`` the longitudes are written east to west, which
    the reader has to reverse along with every field.
    """
    model = ERA5()
    nlev = model._levels
    lons = _LONS_ASC[::-1] if lons_descending else _LONS_ASC
    nlat, nlon = len(_LATS_DESC), len(lons)

    terrain = _terrain(_LATS_DESC[:, None], lons[None, :])
    z_surface = terrain * _g0

    t = np.full((nlev, nlat, nlon), 280.0)
    q = np.full((nlev, nlat, nlon), 1e-3)

    # surface pressure consistent with the terrain (barometric)
    psurf = 101325.0 * np.exp(-terrain / 8400.0)
    lnsp = np.log(psurf)

    # calcgeoh turns its t argument into virtual temperature in place, so pass
    # a copy so the synthetic file carries the temperature CDS puts in its raw
    # response. (Files written by the real fetch currently store the mutated
    # array instead — a separate defect; this fixture writes the intended
    # layout.)
    z_full, _, _ = calcgeoh(
        lnsp=lnsp,
        t=t.copy(),
        q=q,
        z_surface=z_surface,
        a=model._a,
        b=model._b,
        R_d=model._R_d,
        num_levels=nlev,
    )

    dims = ('valid_time', 'model_level', 'latitude', 'longitude')
    data_vars = {
        't': (dims, t[np.newaxis]),
        'q': (dims, q[np.newaxis]),
        'z': (dims, np.asarray(z_full)[np.newaxis]),
        # the reader takes lnsp as [0, 0], so it must carry both leading dims
        'lnsp': (dims, np.broadcast_to(lnsp, (1, nlev, nlat, nlon)).copy()),
    }
    if include_z_surface:
        # like lnsp, read back by the reader as [0, 0]
        data_vars['z_surface'] = (dims, np.broadcast_to(z_surface, (1, nlev, nlat, nlon)).copy())
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'valid_time': [0],
            'model_level': np.arange(1, nlev + 1),
            'latitude': _LATS_DESC,
            'longitude': lons,
        },
    )
    ds.to_netcdf(path)
    return model


@pytest.fixture(
    params=[(False, False), (True, False), (False, True), (True, True)],
    ids=['legacy_layout', 'with_z_surface', 'legacy_layout-lons_descending', 'with_z_surface-lons_descending'],
)
def loaded_model(tmp_path, request):
    """The loaded synthetic model, in both file layouts and both longitude orders.

    The legacy layout exercises the lowest-full-level fallback (and with it
    the orientation of the z cube); the z_surface layout exercises the
    orientation of the stored surface field. The descending-longitude variants
    exercise the longitude reversal of every field.
    """
    include_z_surface, lons_descending = request.param
    f = tmp_path / 'ERA-5_synthetic_ml.nc'
    model = _write_synthetic_ml_file(f, include_z_surface=include_z_surface, lons_descending=lons_descending)
    model._ll_bounds = (32.5, 36.5, -120.5, -117.5)
    model._load_model_level(f)
    return model


def test_model_level_latitudes_are_ascending(loaded_model):
    """The reader is expected to flip ECMWF's descending latitudes."""
    lats = loaded_model._lats[:, 0]
    assert lats[0] < lats[-1]
    assert np.allclose(np.sort(lats), lats)


def test_model_level_longitudes_are_ascending(loaded_model):
    """The reader is expected to flip descending longitudes, and to leave ascending ones alone."""
    lons = loaded_model._lons[0, :]
    assert lons[0] < lons[-1]
    assert np.allclose(np.sort(lons), lons)


def test_surface_height_matches_terrain_per_column(loaded_model):
    """Heights must land on the column they came from, not a mirror of it.

    Reversing only the level axis of z (the bug this guards) leaves the height
    cube mirrored in latitude, so the 3000 m row shows up at 33 N. Missing the
    longitude reversal of one field (or applying it to the wrong one) mirrors
    the 200 m east-west ramp.
    """
    m = loaded_model
    surface_height = m._zs[..., 0]  # _zs runs bottom -> top after loading

    expected = _terrain(m._lats, m._lons)
    # geo_to_ht applies a small geometric correction, so allow a loose tolerance;
    # the failure modes being guarded against are off by 200-3000 m.
    assert np.allclose(surface_height, expected, atol=50.0)


def test_surface_height_and_pressure_are_consistent(loaded_model):
    """High terrain must carry low surface pressure.

    This is the single most diagnostic invariant: with z reversed on the wrong
    axis the correlation goes negative, because heights come from the mirrored
    latitude while pressure does not.
    """
    m = loaded_model
    surface_height = m._zs[..., 0].ravel()
    surface_pressure = m._p[..., 0].ravel()

    corr = np.corrcoef(surface_height, -surface_pressure)[0, 1]
    assert corr > 0.99, f'height/pressure correlation {corr:.4f}; z is likely mirrored in latitude'


def test_heights_increase_upward(loaded_model):
    """_zs is ordered surface -> top of atmosphere after loading."""
    m = loaded_model
    assert (np.diff(m._zs, axis=2) > 0).all()
    assert (np.diff(m._p, axis=2) < 0).all()


def test_z_surface_starts_columns_at_the_terrain(tmp_path):
    """With z_surface stored, the reader reproduces the fetch-time heights exactly.

    The reader re-runs the hydrostatic integration from the surface
    geopotential it is given. Started from the true surface, it lands each
    full level exactly where the fetch-time calcgeoh put it in the stored z
    cube; started from the lowest full level (the legacy fallback), every
    column comes out ~10 m too high; started from a mirrored z_surface, the
    error would be the terrain difference between mirrored latitudes
    (hundreds to thousands of meters). A tight per-column tolerance catches
    all of these.
    """
    f = tmp_path / 'ERA-5_synthetic_ml_z_surface.nc'
    model = _write_synthetic_ml_file(f, include_z_surface=True)
    model._ll_bounds = (32.5, 36.5, -120.5, -117.5)
    model._load_model_level(f)

    with xr.open_dataset(f) as ds:
        z_cube = ds['z'].values[0]  # (level, lat, lon), latitude descending

    # stored cube runs top -> surface with descending latitude; _zs runs
    # surface -> top with ascending latitude
    expected_lowest = geo_to_ht(model._lats, z_cube[-1, ::-1] / _g0)
    assert np.allclose(model._zs[..., 0], expected_lowest, atol=1e-3)
