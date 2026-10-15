"""Regression tests for the full-level pressure returned by `calcgeoh`.

ECMWF hybrid coordinates define pressure on HALF levels:

    p_half[k] = a[k] + b[k] * p_surface

The value belonging to a FULL level is the mean of the two half levels that
bracket it. `calcgeoh` pairs the pressure it returns with `geopotential[idx]`,
which is a full-level height, so it must return the full-level pressure.

From 2019 until 2021-06-30 (`fc7d0f59`) the averaging was missing; it was
restored, then commented out six months later inside a file move (`459763124`)
and replayed onto `dev` in `44ebc41b`. Every release except v0.1.0 has shipped
the half-level-above value.

The consequence is not subtle. Each full level is handed a pressure that
belongs roughly half a model layer higher -- about 145 m through the mid
troposphere -- so every level is biased low and the integrated hydrostatic
delay with it, by ~4 cm at sea level. `test_half_level_pressure_biases_the_delay`
quantifies that from the coefficients alone, and is self-contained: it needs no
weather file and no network.
"""
import numpy as np

from RAiDER.models.era5 import ERA5
from RAiDER.utilFcns import calcgeoh


_LATS = np.array([40.0, 39.0])
_LONS = np.array([250.0, 251.0])
_P_SURFACE = 101325.0
_K1 = 0.776  # K/Pa, the hydrostatic refractivity coefficient used by all models


def _uniform_column(p_surface=_P_SURFACE, t_value=280.0):
    """A flat, isothermal grid — enough to exercise the pressure assignment."""
    model = ERA5()
    nlev = model._levels
    nlat, nlon = len(_LATS), len(_LONS)

    t = np.full((nlev, nlat, nlon), t_value)
    q = np.full((nlev, nlat, nlon), 1e-3)
    lnsp = np.full((nlat, nlon), np.log(p_surface))
    z_surface = np.zeros((nlat, nlon))

    z, p, hgt = calcgeoh(
        lnsp=lnsp, t=t, q=q, z_surface=z_surface,
        a=model._a, b=model._b, R_d=model._R_d, num_levels=nlev,
    )
    return model, np.asarray(p), np.asarray(hgt)


def test_returned_pressure_is_the_full_level_value():
    """Each full level must get the MEAN of its bracketing half levels.

    This is the direct statement of the defect: returning `ph_lev` alone hands
    every level the half-level pressure ABOVE it. Asserted exactly, because the
    relation is exact — it does not depend on the atmosphere at all.
    """
    model, p, _ = _uniform_column()
    a = np.asarray(model._a, dtype=float)
    b = np.asarray(model._b, dtype=float)

    expected = 0.5 * (a[:-1] + a[1:]) + 0.5 * (b[:-1] + b[1:]) * _P_SURFACE

    assert p.shape[0] == expected.size
    np.testing.assert_allclose(p[:, 0, 0], expected, rtol=1e-12)


def test_returned_pressure_is_not_the_half_level_above():
    """Guard the specific regression, so a revert cannot pass silently."""
    model, p, _ = _uniform_column()
    a = np.asarray(model._a, dtype=float)
    b = np.asarray(model._b, dtype=float)

    half_level_above = a[:-1] + b[:-1] * _P_SURFACE
    gap = np.abs(p[:, 0, 0] - half_level_above)

    # the two differ by half a layer's pressure thickness, which peaks at
    # ~10 hPa in the mid troposphere
    assert gap.max() > 100.0, 'pressure looks like the half level above, not the full level'


def test_half_level_pressure_biases_the_delay():
    """Quantify what the missing average costs, from the coefficients alone.

    Integrates k1 * p / T over the column twice — once with the correct
    full-level pressure and once with the half level above — on the heights
    `calcgeoh` itself returns. Self-contained: no weather file, no network.
    """
    model, p_full, hgt = _uniform_column()
    a = np.asarray(model._a, dtype=float)
    b = np.asarray(model._b, dtype=float)
    t_value = 280.0

    p_half_above = a[:-1] + b[:-1] * _P_SURFACE

    # calcgeoh returns levels top -> bottom; integrate over increasing height
    z = np.asarray(hgt)[:, 0, 0][::-1]
    order = np.argsort(z)
    z = z[order]

    def ztd(pressure):
        n = 1e-6 * _K1 * pressure[::-1][order] / t_value
        # explicit trapezoid: np.trapz was renamed np.trapezoid in numpy 2
        return float(np.sum(0.5 * (n[:-1] + n[1:]) * np.diff(z)))

    deficit_mm = (ztd(p_half_above) - ztd(p_full[:, 0, 0])) * 1000.0

    # the sign is what matters: the half-level-above value is always lower, so
    # the delay comes out too SMALL. Magnitude is ~4 cm on a sea-level column.
    assert deficit_mm < 0
    assert 20.0 < abs(deficit_mm) < 80.0, f'expected a few cm, got {deficit_mm:.1f} mm'
