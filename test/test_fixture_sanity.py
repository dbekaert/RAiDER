"""Cheap physical sanity checks on every processed weather-model fixture.

The golden-value tests read precomputed fields straight out of these files, so a
corrupted fixture can sit in the repository unnoticed as long as no golden value
happens to sample the bad columns. These checks look at every column instead.
They are deliberately loose: they catch broken files, not small biases.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from test import TEST_DIR


PROCESSED_VARS = ('t', 'p', 'e', 'wet', 'hydro', 'wet_total', 'hydro_total')


def _processed_fixtures() -> list[Path]:
    """Every netCDF under test/ that holds a processed weather model."""
    found = []
    for path in sorted(TEST_DIR.rglob('*.nc')):
        with xr.open_dataset(path) as ds:
            if {'t', 'hydro_total', 'wet_total'} <= set(ds.data_vars):
                found.append(path)
    return found


FIXTURES = _processed_fixtures()


@pytest.fixture(scope='module', params=FIXTURES, ids=[str(p.relative_to(TEST_DIR)) for p in FIXTURES])
def fixture_ds(request: pytest.FixtureRequest) -> xr.Dataset:
    with xr.open_dataset(request.param) as ds:
        yield ds.load()


def test_processed_fixtures_were_found() -> None:
    assert len(FIXTURES) >= 3


def test_no_sentinel_values_below_the_top_level(fixture_ds: xr.Dataset) -> None:
    """Fill values are only expected at the top level of the height grid."""
    for var in PROCESSED_VARS:
        values = fixture_ds[var].values[:-1]
        bad = ~np.isfinite(values) | (np.abs(values) > 1e10)
        assert not bad.any(), f'{var}: {int(bad.sum())} non-finite or fill values below the top level'


def test_temperature_is_physical_below_20_km(fixture_ds: xr.Dataset) -> None:
    t = fixture_ds['t'].where(fixture_ds['z'] < 20_000, drop=True).values
    assert t.min() >= 150, f'minimum temperature {t.min():.1f} K'
    assert t.max() <= 340, f'maximum temperature {t.max():.1f} K'


def test_no_column_is_colder_at_sea_level_than_at_10_km(fixture_ds: xr.Dataset) -> None:
    """Catches columns whose profile is scrambled even when every value is in range."""
    t = fixture_ds['t'].interp(z=[0.0, 10_000.0]).values
    inverted = t[0] <= t[1]
    assert not inverted.any(), f'{int(inverted.sum())} of {inverted.size} columns colder at 0 m than at 10 km'


def test_sea_level_zenith_delay_is_plausible(fixture_ds: xr.Dataset) -> None:
    ztd = (fixture_ds['hydro_total'] + fixture_ds['wet_total']).interp(z=0.0).values
    assert ztd.min() >= 2.0, f'minimum ZTD at 0 m is {ztd.min():.3f} m'
    assert ztd.max() <= 2.8, f'maximum ZTD at 0 m is {ztd.max():.3f} m'
