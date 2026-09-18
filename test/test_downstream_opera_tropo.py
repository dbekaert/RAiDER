"""Downstream contract test: OPERA TROPO's use of the HRES model-level path.

WHY THIS LIVES IN RAiDER
------------------------
OPERA L4 TROPO-ZENITH (`opera-adt/tropo`) pins a RAiDER release and drives the
HRES model-level path directly -- `calcgeoh`, `_get_heights`, `_find_e`,
`_uniform_in_z`, the refractivity getters and `_getZTD` -- rather than going
through `calcDelays`. Those are internal entry points, so nothing in RAiDER's
own suite covers that sequence, and a change to any of them moves an
operational product without anything here going red.

This test is the tripwire. It is not asking for those internals to be frozen:
refactor them freely. It asks that the delay they produce stays physically
correct, so that a PR which changes what OPERA would get says so in CI.

WHAT IT PINS
------------
An INDEPENDENT reference, not a recording of RAiDER's own past output:
Saastamoinen's closed form for the zenith hydrostatic delay, a function of
surface pressure and latitude alone.

    ZHD = 0.0022768 * p_s[hPa] / (1 - 0.00266*cos(2*phi) - 0.00028*h_km)

A golden file regenerated from a broken run defends the breakage; a closed form
cannot be regenerated. On the fixture below, with the half-level average
restored, RAiDER lands at 1.55 mm RMS; without it, at 31.5 mm, biased -23.9 mm
and ramping +4.09 mm per km of terrain.

THE FIXTURE
-----------
`opera_tropo_data/hres_columns_9.npz`, 8 KB: nine HRES columns on the operational 137-level
grid, chosen to span sea level to 6.3 km terrain (surface pressure 1020 down to
469 hPa) so the terrain dependence is exercised. They are a curated sample
reshaped to a 3x3 grid, not a contiguous scene -- every calculation here is
column-wise, so that is sufficient and keeps the fixture small enough to commit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from RAiDER.models.hres import HRES


DATA = Path(__file__).parent / 'opera_tropo_data' / 'hres_columns_9.npz'

#: Measured on this fixture, on this branch: 1.55 mm RMS / -1.37 mm bias /
#: +0.17 mm per km. Without the half-level average: 33.1 / -23.9 / +4.70.
#: The thresholds sit clear of the first and well under the second, so they
#: neither flap nor pass the defect. Widen them only with a measurement to
#: justify the new number.
ZHD_RMS_TOLERANCE_MM = 5.0
ZHD_BIAS_TOLERANCE_MM = 3.0
ZHD_SLOPE_TOLERANCE_MM_PER_KM = 1.5


def saastamoinen_zhd(p_surface_hpa, lat_deg, height_m):
    """Zenith hydrostatic delay in metres. Davis et al. (1985), eq. A11.

    The 0.0022768 coefficient is metres per hPa, not per Pa.
    """
    phi = np.deg2rad(lat_deg)
    denom = 1 - 0.00266 * np.cos(2 * phi) - 0.00028 * (height_m / 1000.0)
    return 0.0022768 * p_surface_hpa / denom


def _run_opera_tropo_path(fixture):
    """Drive the HRES sequence exactly as opera-adt/tropo does.

    Kept deliberately literal rather than factored, so a reviewer can compare it
    line for line against `tropo/src/opera_tropo` and see that this is really
    what the downstream project calls.
    """
    model = HRES()
    model._t = fixture['t'].astype(float)
    model._q = fixture['q'].astype(float)
    model._lons = fixture['lons'].astype(float)
    model._lats = fixture['lats'].astype(float)

    # NOTE 1: this signature changed after v0.5.3 -- it used to read t/q off
    # the model instance. opera-adt/tropo still calls the two-argument form, so
    # it will raise TypeError on upgrade until it is updated.
    #
    # NOTE 2: t is passed as a COPY on purpose. calcgeoh converts temperature
    # to virtual temperature with an in-place multiply (issue #819), so the
    # caller's array comes back inflated by up to 0.65 % and every later step
    # -- _find_e, both refractivity getters -- then sees moisture-inflated
    # temperatures. That is a real defect and a real consequence for callers,
    # but it is not the one this PR fixes. Copying isolates the pressure path,
    # so these assertions measure one thing. On this fixture the mutation
    # costs an extra 1.7 mm RMS (3.26 vs 1.55) and +0.6 mm/km of slope; when
    # #819 lands, drop the .copy() and the numbers should not move.
    _, pressure, geo_height = model._calculategeoh(
        fixture['z_surface'].astype(float),
        fixture['lnsp'].astype(float),
        model._t.copy(),
        model._q,
    )
    model._p = pressure
    model._get_heights(model._lats, geo_height.transpose(1, 2, 0))
    heights = model._zs.copy()

    # (level, lat, lon) -> (lat, lon, height), bottom-up
    model._p = np.flip(model._p.transpose(1, 2, 0), axis=2)
    model._t = np.flip(model._t.transpose(1, 2, 0), axis=2)
    model._q = np.flip(model._q.transpose(1, 2, 0), axis=2)
    model._zs = np.flip(heights, axis=2)
    model._xs, model._ys = model._lons.copy(), model._lats.copy()

    model._find_e()
    model._uniform_in_z(_zlevels=None)
    model._checkForNans()
    model._get_wet_refractivity()
    model._get_hydro_refractivity()
    model._getZTD()

    return np.asarray(model._hydrostatic_ztd), np.asarray(model._zs)


@pytest.fixture
def zhd_residual_mm():
    """RAiDER's ZHD at each column's own surface, minus Saastamoinen."""
    if not DATA.exists():
        pytest.skip(f'missing fixture {DATA}')
    fixture = dict(np.load(DATA))

    zhd, zs = _run_opera_tropo_path(fixture)

    terrain = fixture['terrain'].astype(float)
    p_surface_hpa = np.exp(fixture['lnsp'].astype(float)) / 100.0
    lats = fixture['lats'].astype(float)

    ny, nx = terrain.shape
    at_surface = np.empty((ny, nx))
    for i in range(ny):
        for j in range(nx):
            at_surface[i, j] = np.interp(terrain[i, j], zs, zhd[i, j, :])

    reference = saastamoinen_zhd(p_surface_hpa, lats, terrain)
    return (at_surface - reference) * 1000.0, terrain


def test_opera_tropo_zhd_matches_saastamoinen(zhd_residual_mm):
    """The delay OPERA TROPO would get must match the closed form."""
    diff_mm, _ = zhd_residual_mm
    rms = float(np.sqrt(np.nanmean(diff_mm ** 2)))

    assert rms < ZHD_RMS_TOLERANCE_MM, (
        f'ZHD differs from Saastamoinen by {rms:.1f} mm RMS on the HRES '
        f'model-level path. This path is what OPERA L4 TROPO-ZENITH runs, so a '
        f'change here moves an operational product. Verify the change is '
        f'intended, then re-measure and update the tolerance in this file.'
    )


def test_opera_tropo_zhd_has_no_systematic_bias(zhd_residual_mm):
    """A fractional column error shows up as a one-signed offset."""
    diff_mm, _ = zhd_residual_mm
    bias = float(np.nanmedian(diff_mm))

    assert abs(bias) < ZHD_BIAS_TOLERANCE_MM, (
        f'ZHD is biased {bias:+.1f} mm against Saastamoinen. Tens of mm, all '
        f'one sign, is the signature of a pressure paired with the wrong model '
        f'level rather than of interpolation noise.'
    )


def test_opera_tropo_zhd_does_not_ramp_with_terrain(zhd_residual_mm):
    """The residual must not scale with column length.

    The slope is the diagnostic, not the correlation: Saastamoinen carries its
    own approximate height term, so even a correct RAiDER correlates strongly
    with terrain -- but at -0.4 mm/km, not +3.6.
    """
    diff_mm, terrain = zhd_residual_mm
    h_km = terrain.ravel() / 1000.0
    d = diff_mm.ravel()
    ok = np.isfinite(d) & np.isfinite(h_km)

    slope = float(np.polyfit(h_km[ok], d[ok], 1)[0])
    assert abs(slope) < ZHD_SLOPE_TOLERANCE_MM_PER_KM, (
        f'ZHD residual ramps {slope:+.2f} mm per km of terrain, so the error is '
        f'proportional to the column. High-terrain OPERA tiles would be worst '
        f'affected.'
    )


def test_report_delta_against_opera_production(zhd_residual_mm, capsys):
    """Report how far this build moves OPERA's shipped numbers. Never fails.

    The assertions above are the contract: they say the delay is physically
    right, and no one can regenerate them into passing. This one is a notice
    board. It prints the per-column difference against what OPERA L4
    TROPO-ZENITH ships today so a reviewer can see, in CI, that a change
    reaches an operational product and by how much.

    It deliberately does not assert. Whether a given delta is acceptable
    depends on reprocessing cost and product requirements that live in
    `opera-adt/tropo`, not here -- and a threshold a RAiDER reviewer cannot
    evaluate is one they would raise until it stopped complaining. The
    corresponding hard check belongs downstream, pinned by the people who own
    the product.
    """
    baseline_path = DATA.parent / 'opera_production_zhd.json'
    if not baseline_path.exists():
        pytest.skip(f'missing baseline {baseline_path}')
    baseline = json.loads(baseline_path.read_text())

    fixture = dict(np.load(DATA))
    zhd, zs = _run_opera_tropo_path(fixture)
    terrain = fixture['terrain'].astype(float)

    here, shipped = [], baseline['production_zhd_m']
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            here.append(float(np.interp(terrain[i, j], zs, zhd[i, j, :])))

    delta_mm = np.array([(a - b) * 1000.0 for a, b in zip(here, shipped)])
    heights = np.array(baseline['terrain_m'])

    lines = [
        '',
        'OPERA L4 TROPO-ZENITH impact of this build (informational, never fails):',
        f'  {"terrain (m)":>12s} {"shipped (mm)":>14s} {"this build":>12s} {"delta":>9s}',
    ]
    for h, s, d in zip(heights, shipped, delta_mm):
        lines.append(f'  {h:12.0f} {s * 1000:14.2f} {s * 1000 + d:12.2f} {d:+9.2f}')
    lines.append(
        f'  median {np.median(delta_mm):+.2f} mm, range {delta_mm.min():+.2f} to '
        f'{delta_mm.max():+.2f} mm over {heights.min():.0f}-{heights.max():.0f} m terrain.'
    )
    lines.append(
        '  A non-zero delta is expected on this branch: it is the calcgeoh fix. '
        'If you did not intend to move these numbers, that is the finding.'
    )
    with capsys.disabled():
        print('\n'.join(lines))
