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
import numpy as np
import pytest
import xarray as xr

from RAiDER.constants import _g0
from RAiDER.models.era5 import ERA5
from RAiDER.utilFcns import calcgeoh


# Terrain rising steeply from south to north, so that a latitude reflection is
# unambiguous: the southern rows are at sea level, the northern rows are high.
_LATS_DESC = np.array([36.0, 35.0, 34.0, 33.0])   # ECMWF order: descending
_LONS = np.array([240.0, 241.0, 242.0])           # 0-360, as ECMWF delivers
_TERRAIN = {36.0: 3000.0, 35.0: 2000.0, 34.0: 1000.0, 33.0: 0.0}


def _write_synthetic_ml_file(path):
    """Write a minimal ERA-5 model-level file in ECMWF's native ordering.

    Mirrors what RAiDER's own _fetch writes: z is the *full* geopotential cube
    (computed with calcgeoh), not the surface-only field CDS returns.
    """
    model = ERA5()
    nlev = model._levels
    nlat, nlon = len(_LATS_DESC), len(_LONS)

    terrain = np.array([_TERRAIN[la] for la in _LATS_DESC])[:, None] * np.ones((nlat, nlon))
    z_surface = terrain * _g0

    t = np.full((nlev, nlat, nlon), 280.0)
    q = np.full((nlev, nlat, nlon), 1e-3)

    # surface pressure consistent with the terrain (barometric)
    psurf = 101325.0 * np.exp(-terrain / 8400.0)
    lnsp = np.log(psurf)

    z_full, _, _ = calcgeoh(
        lnsp=lnsp, t=t, q=q, z_surface=z_surface,
        a=model._a, b=model._b, R_d=model._R_d, num_levels=nlev,
    )

    dims = ('valid_time', 'model_level', 'latitude', 'longitude')
    ds = xr.Dataset(
        data_vars={
            't': (dims, t[np.newaxis]),
            'q': (dims, q[np.newaxis]),
            'z': (dims, np.asarray(z_full)[np.newaxis]),
            # the reader takes lnsp as [0, 0], so it must carry both leading dims
            'lnsp': (dims, np.broadcast_to(lnsp, (1, nlev, nlat, nlon)).copy()),
        },
        coords={
            'valid_time': [0],
            'model_level': np.arange(1, nlev + 1),
            'latitude': _LATS_DESC,
            'longitude': _LONS,
        },
    )
    ds.to_netcdf(path)
    return model


@pytest.fixture
def loaded_model(tmp_path):
    f = tmp_path / 'ERA-5_synthetic_ml.nc'
    model = _write_synthetic_ml_file(f)
    model._ll_bounds = (32.5, 36.5, -120.5, -117.5)
    model._load_model_level(f)
    return model


def test_model_level_latitudes_are_ascending(loaded_model):
    """The reader is expected to flip ECMWF's descending latitudes."""
    lats = loaded_model._lats[:, 0]
    assert lats[0] < lats[-1]
    assert np.allclose(np.sort(lats), lats)


def test_surface_height_matches_terrain_per_latitude(loaded_model):
    """Heights must land on the latitude they came from, not its mirror.

    Reversing only the level axis of z (the bug this guards) leaves the height
    cube mirrored in latitude, so the 3000 m row shows up at 33 N.
    """
    m = loaded_model
    lats = m._lats[:, 0]
    surface_height = m._zs[..., 0]   # _zs runs bottom -> top after loading

    expected = np.array([_TERRAIN[round(float(la))] for la in lats])
    # geo_to_ht applies a small geometric correction, so allow a loose tolerance;
    # the failure mode being guarded against is off by ~1000-3000 m.
    assert np.allclose(surface_height.mean(axis=1), expected, atol=50.0)


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
