"""Unit tests for ECMWF/ERA5/ERA5T weather model loaders.

These tests exercise the loading and axis-ordering logic that was broken
in the multiday_download branch (ecmwf.py commit 3b64ad3..d734fd7).  The
primary failure mode was _load_pressure_level dividing geopotential heights
by g0 a second time and applying axis flips to the wrong dimensions,
producing hydrostatic ZTDs of ~1 m instead of ~2.3 m.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from RAiDER.models.era5 import ERA5
from RAiDER.models.era5t import ERA5T


# ---------------------------------------------------------------------------
# Helpers to build synthetic netCDF files
# ---------------------------------------------------------------------------

G0 = 9.80665  # m/s²

# Small but physically plausible grid
_NLAT = 4
_NLON = 5
_NLEV = 6

# Lats: N→S descending, as CDS returns them; loader must flip to S→N.
_LATS_DESC = np.array([52.0, 51.0, 50.0, 49.0])
_LATS_ASC = _LATS_DESC[::-1]

# Lons: ascending, centred on Europe
_LONS = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

# Pressure levels in hPa, surface-first (1000 → 10), as the batch writer
# produces them from a standard CDS download.
_LEVS_HPA_DESC = np.array([1000.0, 500.0, 200.0, 100.0, 50.0, 10.0])
_LEVS_PA_DESC = _LEVS_HPA_DESC * 100.0

# Geopotential heights (m) matching the pressure levels above, surface first.
# Values are broadly consistent with the standard atmosphere.
_Z_GHT_SURFACE_FIRST = np.array([100.0, 5_800.0, 11_800.0, 16_200.0, 20_600.0, 31_100.0])


def _make_t(nlat=_NLAT, nlon=_NLON, nlev=_NLEV) -> np.ndarray:
    """Return a (nlat, nlon, nlev) temperature array in K (surface first in lev)."""
    sfc_t = 290.0
    lapse = np.linspace(0, 80, nlev)  # decreases with altitude
    return np.broadcast_to(sfc_t - lapse, (nlat, nlon, nlev)).copy().astype(np.float32)


def _make_q(nlat=_NLAT, nlon=_NLON, nlev=_NLEV) -> np.ndarray:
    """Return a (nlat, nlon, nlev) specific humidity array (surface first in lev)."""
    q_sfc = 0.01
    decay = np.exp(-np.linspace(0, 5, nlev))
    return np.broadcast_to(q_sfc * decay, (nlat, nlon, nlev)).copy().astype(np.float32)


def _make_z_ght(nlat=_NLAT, nlon=_NLON) -> np.ndarray:
    """Return a (nlat, nlon, nlev) geopotential-height array, surface first."""
    return np.broadcast_to(_Z_GHT_SURFACE_FIRST, (nlat, nlon, _NLEV)).copy().astype(np.float64)


def write_pl_file_batch_format(
    path: Path,
    *,
    lats: np.ndarray = _LATS_DESC,
    lons: np.ndarray = _LONS,
    levs_hpa: np.ndarray = _LEVS_HPA_DESC,
    z_ght: np.ndarray | None = None,
    time_dim_name: str = 'valid_time',
    lev_dim_name: str = 'pressure_level',
    z_as_geopotential: bool = False,
) -> Path:
    """Write a synthetic pressure-level file in the format _batch_get_from_cds produces.

    * dims (time, lat, lon, lev) with z in geopotential-height metres (< 50 km).
    * Optionally multiply z by g0 to produce a raw-geopotential file.
    * ``time_dim_name`` and ``lev_dim_name`` let us probe the alternate coord names.
    """
    nlat, nlon, nlev = len(lats), len(lons), len(levs_hpa)
    if z_ght is None:
        z_ght = np.broadcast_to(_Z_GHT_SURFACE_FIRST[:nlev], (nlat, nlon, nlev)).copy()

    t = _make_t(nlat, nlon, nlev)
    q = _make_q(nlat, nlon, nlev)
    z_out = z_ght * G0 if z_as_geopotential else z_ght.copy()

    t4d = t[np.newaxis]      # (1, nlat, nlon, nlev)
    q4d = q[np.newaxis]
    z4d = z_out[np.newaxis]

    time_val = np.array(['2020-01-01'], dtype='datetime64[ns]')

    # Match what the real writers record: the batch writer stores geopotential
    # height in metres, a raw CDS file stores geopotential in m**2 s**-2.
    z_attrs = {'units': 'm**2 s**-2'} if z_as_geopotential else {'units': 'm'}

    ds = xr.Dataset(
        {
            'z': xr.Variable((time_dim_name, 'latitude', 'longitude', lev_dim_name), z4d, z_attrs),
            't': xr.Variable((time_dim_name, 'latitude', 'longitude', lev_dim_name), t4d),
            'q': xr.Variable((time_dim_name, 'latitude', 'longitude', lev_dim_name), q4d),
        },
        coords={
            time_dim_name: time_val,
            'latitude': lats,
            'longitude': lons,
            lev_dim_name: levs_hpa,
        },
    )
    ds.to_netcdf(path)
    return path


def write_ml_file(path: Path) -> Path:
    """Write a minimal model-level file matching the format _makeDataCubes expects."""
    nlat, nlon, nlev = _NLAT, _NLON, _NLEV
    lats = _LATS_ASC
    lons = _LONS

    t = _make_t(nlat, nlon, nlev).transpose(2, 0, 1)   # (nlev, nlat, nlon)
    q = _make_q(nlat, nlon, nlev).transpose(2, 0, 1)
    z = np.broadcast_to(
        _Z_GHT_SURFACE_FIRST[:, np.newaxis, np.newaxis],
        (nlev, nlat, nlon),
    ).copy().astype(np.float32)
    lnsp = np.full((nlat, nlon), np.log(95_000.0), dtype=np.float32)

    ds = xr.Dataset(
        {
            't':    xr.Variable(('valid_time', 'model_level', 'latitude', 'longitude'), t[np.newaxis]),
            'q':    xr.Variable(('valid_time', 'model_level', 'latitude', 'longitude'), q[np.newaxis]),
            'z':    xr.Variable(('valid_time', 'model_level', 'latitude', 'longitude'), z[np.newaxis]),
            'lnsp': xr.Variable(('valid_time', 'sfc_level', 'latitude', 'longitude'), lnsp[np.newaxis, np.newaxis]),
        },
        coords={
            'valid_time': np.array(['2020-01-01'], dtype='datetime64[ns]'),
            'model_level': np.arange(1, nlev + 1),
            'latitude': lats,
            'longitude': lons,
        },
    )
    ds.to_netcdf(path)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def era5():
    return ERA5()


@pytest.fixture()
def era5t():
    return ERA5T()


# ---------------------------------------------------------------------------
# Init / attribute tests
# ---------------------------------------------------------------------------

class TestERA5Init:
    def test_default_level_type_is_ml(self, era5: ERA5) -> None:
        assert era5._model_level_type == 'ml'

    def test_name(self, era5: ERA5) -> None:
        assert era5._Name == 'ERA-5'

    def test_humidity_type(self, era5: ERA5) -> None:
        assert era5._humidityType == 'q'

    def test_projection(self, era5: ERA5) -> None:
        assert era5._proj.to_epsg() == 4326

    def test_valid_range_start(self, era5: ERA5) -> None:
        assert era5._valid_range[0] == dt.datetime(1950, 1, 1, tzinfo=dt.timezone.utc)

    def test_load_weather_dispatches_to_pl(self, era5: ERA5, tmp_path: Path) -> None:
        """load_weather should call _load_pressure_level for pl type."""
        f = write_pl_file_batch_format(tmp_path / 'test.nc')
        era5.setLevelType('pl')
        era5.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])
        era5.files = [f]

        era5.load_weather(f)

        assert era5._t is not None
        assert era5._q is not None


class TestERA5TInit:
    def test_name(self, era5t: ERA5T) -> None:
        assert era5t._Name == 'ERA-5T'

    def test_expver(self, era5t: ERA5T) -> None:
        assert era5t._expver == '0005'

    def test_dataset(self, era5t: ERA5T) -> None:
        assert era5t._dataset == 'era5t'

    def test_inherits_ml_level_type(self, era5t: ERA5T) -> None:
        assert era5t._model_level_type == 'ml'

    def test_lag_time_is_one_day(self, era5t: ERA5T) -> None:
        assert era5t._lag_time == dt.timedelta(days=1)

    def test_valid_range_extends_to_now(self, era5t: ERA5T) -> None:
        # ERA5T covers up to "now"; check the upper bound is in the future relative to
        # a known past date
        past = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
        assert era5t._valid_range[1] > past


# ---------------------------------------------------------------------------
# _load_pressure_level tests
# ---------------------------------------------------------------------------

class TestLoadPressureLevel:
    """All variants of _load_pressure_level — axis ordering, unit guards, coord names."""

    def _setup_and_load(self, path: Path, era5: ERA5) -> None:
        era5.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])
        era5._load_pressure_level(path)

    # --- shape correctness -------------------------------------------------

    def test_output_shapes_are_lat_lon_lev(self, era5: ERA5, tmp_path: Path) -> None:
        f = write_pl_file_batch_format(tmp_path / 'test.nc')
        self._setup_and_load(f, era5)

        assert era5._t.shape == (_NLAT, _NLON, _NLEV)
        assert era5._q.shape == (_NLAT, _NLON, _NLEV)
        assert era5._zs.shape == (_NLAT, _NLON, _NLEV)
        assert era5._p.shape == (_NLAT, _NLON, _NLEV)

    # --- lat/lon ordering --------------------------------------------------

    def test_lats_are_ascending_south_to_north(self, era5: ERA5, tmp_path: Path) -> None:
        """Loader must flip descending CDS lats to ascending."""
        f = write_pl_file_batch_format(tmp_path / 'test.nc', lats=_LATS_DESC)
        self._setup_and_load(f, era5)

        # _lats is a 2D meshgrid; first row should be the southernmost
        assert era5._lats[0, 0] < era5._lats[-1, 0]

    def test_lats_already_ascending_unchanged(self, era5: ERA5, tmp_path: Path) -> None:
        f = write_pl_file_batch_format(tmp_path / 'test.nc', lats=_LATS_ASC)
        self._setup_and_load(f, era5)

        assert era5._lats[0, 0] < era5._lats[-1, 0]

    def test_lons_gt_180_normalized(self, era5: ERA5, tmp_path: Path) -> None:
        """Longitudes > 180 must be shifted to [-180, 180]."""
        lons_wrapped = np.array([350.0, 351.0, 352.0, 353.0, 354.0])
        f = write_pl_file_batch_format(tmp_path / 'test.nc', lons=lons_wrapped)
        # Widen bounds to include these lons
        era5.set_latlon_bounds([48.0, 53.0, -15.0, 0.0])
        era5._load_pressure_level(f)

        assert np.all(era5._lons <= 180.0)
        assert np.all(era5._lons >= -180.0)

    def test_descending_lons_flipped(self, era5: ERA5, tmp_path: Path) -> None:
        lons_desc = np.array([14.0, 13.0, 12.0, 11.0, 10.0])
        f = write_pl_file_batch_format(tmp_path / 'test.nc', lons=lons_desc)
        self._setup_and_load(f, era5)

        assert era5._lons[0, 0] < era5._lons[0, -1]

    # --- level ordering (pressure / height) --------------------------------

    def test_pressure_surface_first(self, era5: ERA5, tmp_path: Path) -> None:
        """p[...,0] must be the largest pressure (surface)."""
        f = write_pl_file_batch_format(tmp_path / 'test.nc')
        self._setup_and_load(f, era5)

        assert era5._p[0, 0, 0] > era5._p[0, 0, -1]

    def test_pressure_toa_last(self, era5: ERA5, tmp_path: Path) -> None:
        f = write_pl_file_batch_format(tmp_path / 'test.nc')
        self._setup_and_load(f, era5)

        # Top-of-atmosphere pressure should be <= 1000 Pa
        assert era5._p[0, 0, -1] <= 1_000.0

    def test_ascending_hpa_levels_are_flipped(self, era5: ERA5, tmp_path: Path) -> None:
        """When CDS returns levels ascending (TOA-first), the loader must flip them."""
        levs_asc = _LEVS_HPA_DESC[::-1]  # 10 → 1000 hPa (TOA first)
        z_toa_first = _Z_GHT_SURFACE_FIRST[::-1]  # heights matching TOA-first levels
        z4d = np.broadcast_to(z_toa_first, (_NLAT, _NLON, _NLEV)).copy()
        f = write_pl_file_batch_format(tmp_path / 'test.nc', levs_hpa=levs_asc, z_ght=z4d)
        self._setup_and_load(f, era5)

        assert era5._p[0, 0, 0] > era5._p[0, 0, -1], 'pressure must be surface-first'

    def test_heights_monotonically_increasing(self, era5: ERA5, tmp_path: Path) -> None:
        """Heights must increase from surface to TOA (index 0 → -1)."""
        f = write_pl_file_batch_format(tmp_path / 'test.nc')
        self._setup_and_load(f, era5)

        dz = np.diff(era5._zs, axis=2)
        assert np.all(dz > 0)

    # --- z unit guard (geopotential vs geopotential-height) ----------------

    def test_geopotential_z_divided_by_g0(self, era5: ERA5, tmp_path: Path) -> None:
        """When z is stored as geopotential (m²/s²) the loader divides by g0."""
        f_ght = write_pl_file_batch_format(tmp_path / 'ght.nc', z_as_geopotential=False)
        f_gp = write_pl_file_batch_format(tmp_path / 'gp.nc', z_as_geopotential=True)

        era5_ght = ERA5()
        era5_ght.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])
        era5_ght._load_pressure_level(f_ght)

        era5_gp = ERA5()
        era5_gp.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])
        era5_gp._load_pressure_level(f_gp)

        # Heights should be (approximately) the same once g0 division is applied
        np.testing.assert_allclose(era5_ght._zs, era5_gp._zs, rtol=1e-4)

    def test_geopotential_height_file_not_double_divided(self, era5: ERA5, tmp_path: Path) -> None:
        """A file with z already in metres must NOT be divided again by g0."""
        f = write_pl_file_batch_format(tmp_path / 'test.nc')
        self._setup_and_load(f, era5)

        # The highest point on our test grid should be ~31 km, not ~3 km
        assert era5._zs.max() > 25_000

    # --- alternate coordinate names ----------------------------------------

    def test_time_dim_named_time(self, era5: ERA5, tmp_path: Path) -> None:
        """Loader must handle 'time' as the time-dimension name."""
        f = write_pl_file_batch_format(tmp_path / 'test.nc', time_dim_name='time')
        self._setup_and_load(f, era5)

        assert era5._t.shape == (_NLAT, _NLON, _NLEV)

    def test_lev_dim_named_level(self, era5: ERA5, tmp_path: Path) -> None:
        """Loader must handle 'level' as the level-dimension name."""
        f = write_pl_file_batch_format(tmp_path / 'test.nc', lev_dim_name='level')
        self._setup_and_load(f, era5)

        assert era5._t.shape == (_NLAT, _NLON, _NLEV)

    # --- crude physics sanity check ----------------------------------------

    def test_hydrostatic_ztd_near_sea_level_is_plausible(self, era5: ERA5, tmp_path: Path) -> None:
        """Integrate k1 * p/t to get a rough hydrostatic ZTD.

        This test would have caught the original bug: the double g0 division
        compressed the atmosphere from ~48 km to ~4.9 km, giving a ZTD
        of ~1 m instead of the expected ~2.2 m.
        """
        f = write_pl_file_batch_format(tmp_path / 'test.nc')
        self._setup_and_load(f, era5)

        # Pick the column closest to sea level
        i, j = np.unravel_index(np.argmin(era5._zs[..., 0]), era5._zs[..., 0].shape)
        zz = era5._zs[i, j]
        pp = era5._p[i, j]
        tt = era5._t[i, j]

        n_hydro = 0.776 * pp / tt * 1e-6
        ztd_h = np.trapezoid(n_hydro, zz)

        assert 1.5 < ztd_h < 3.0, (
            f'Hydrostatic ZTD {ztd_h:.3f} m is outside the expected range [1.5, 3.0] m. '
            'This may indicate an axis-ordering or unit bug.'
        )


# ---------------------------------------------------------------------------
# _makeDataCubes / _load_model_level tests
# ---------------------------------------------------------------------------

class TestMakeDataCubes:
    def test_returns_correct_shapes(self, era5: ERA5, tmp_path: Path) -> None:
        f = write_ml_file(tmp_path / 'ml.nc')
        era5.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])
        era5.setLevelType('ml')

        lats, lons, _, _, t, q, lnsp, z = era5._makeDataCubes(f)

        assert t.shape == (_NLEV, _NLAT, _NLON)
        assert q.shape == (_NLEV, _NLAT, _NLON)
        assert lnsp.shape == (_NLAT, _NLON)
        assert z.shape == (_NLEV, _NLAT, _NLON)
        assert len(lats) == _NLAT
        assert len(lons) == _NLON

    def test_raises_when_mask_excludes_all_data(self, era5: ERA5, tmp_path: Path) -> None:
        """Bounds that exclude all data must raise an error (RuntimeError or ValueError)."""
        f = write_ml_file(tmp_path / 'ml.nc')
        era5.set_latlon_bounds([80.0, 85.0, 90.0, 95.0])
        era5.setLevelType('ml')

        # xarray + netCDF4 raises ValueError when trying to index a 0-sized
        # dimension; _makeDataCubes raises RuntimeError for the explicit check.
        # Either is acceptable — both indicate no usable data.
        with pytest.raises((RuntimeError, ValueError)):
            era5._makeDataCubes(f)


def _mock_calculategeoh(nlat, nlon, nlev):
    """Return a mock for _calculategeoh that produces plausible (nlev, nlat, nlon) arrays.

    _calculategeoh requires num_levels == t.shape[0] (137 for ML mode), so a
    synthetic 6-level file would crash the real implementation.  We mock it to
    keep the ML loader tests fast and self-contained.

    _calculategeoh returns data with model levels numbered from the HIGHEST
    elevation (TOA) to the LOWEST (surface), so index 0 is TOA.  The loader
    then transposes and flips to produce surface-first (lat, lon, lev) arrays.
    """
    # TOA-first ordering: highest height / lowest pressure at index 0
    z_toa_first = _Z_GHT_SURFACE_FIRST[:nlev][::-1]   # [31100, ..., 100]
    p_toa_first = _LEVS_PA_DESC[:nlev][::-1]            # [1000, ..., 100000]
    z_3d = np.broadcast_to(
        z_toa_first[:, np.newaxis, np.newaxis],
        (nlev, nlat, nlon),
    ).copy()
    p_3d = np.broadcast_to(
        p_toa_first[:, np.newaxis, np.newaxis],
        (nlev, nlat, nlon),
    ).copy()
    return z_3d, p_3d, z_3d


class TestLoadModelLevel:
    """Tests for _load_model_level.

    _calculategeoh is mocked here because it requires num_levels == t.shape[0]
    (137 for ERA5 ML mode) which would force an impractically large synthetic file.
    The physics of _calculategeoh are tested separately in test_util.py / calcgeoh.
    """

    def _load(self, era5: ERA5, path: Path) -> None:
        era5.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])
        era5.setLevelType('ml')
        mock_return = _mock_calculategeoh(_NLAT, _NLON, _NLEV)
        with patch.object(era5, '_calculategeoh', return_value=mock_return):
            era5._load_model_level(path)

    def test_output_shapes_are_lat_lon_lev(self, era5: ERA5, tmp_path: Path) -> None:
        f = write_ml_file(tmp_path / 'ml.nc')
        self._load(era5, f)

        assert era5._t.shape == (_NLAT, _NLON, _NLEV)
        assert era5._q.shape == (_NLAT, _NLON, _NLEV)
        assert era5._zs.shape == (_NLAT, _NLON, _NLEV)
        assert era5._p.shape == (_NLAT, _NLON, _NLEV)

    def test_heights_monotonically_increasing(self, era5: ERA5, tmp_path: Path) -> None:
        f = write_ml_file(tmp_path / 'ml.nc')
        self._load(era5, f)

        dz = np.diff(era5._zs, axis=2)
        assert np.all(dz > 0)

    def test_pressure_surface_first(self, era5: ERA5, tmp_path: Path) -> None:
        f = write_ml_file(tmp_path / 'ml.nc')
        self._load(era5, f)

        assert era5._p[0, 0, 0] > era5._p[0, 0, -1]


# ---------------------------------------------------------------------------
# batch_fetch tests (ERA5)
# ---------------------------------------------------------------------------

class TestBatchFetch:
    def test_downloads_an_existing_file_when_asked_to(self, era5: ERA5, tmp_path: Path) -> None:
        """batch_fetch downloads what it is given, existing file or not.

        Only the caller (processWM.batch_download_weather_model) knows whether a file
        on disk actually covers the requested bounds and whether force_download was
        set. batch_fetch second-guessing that with its own exists() check is what made
        force_download a no-op.
        """
        existing = tmp_path / 'already_there.nc'
        existing.touch()

        era5.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])

        with patch.object(era5, '_batch_get_from_cds') as mock_batch:
            era5.batch_fetch([(dt.datetime(2020, 1, 1, 0), existing)])

        mock_batch.assert_called_once()
        assert [p for _, p in mock_batch.call_args[0][0]] == [existing]

    def test_queues_missing_file(self, era5: ERA5, tmp_path: Path) -> None:
        """A file that does not exist must be passed to _batch_get_from_cds."""
        missing = tmp_path / 'missing.nc'

        era5.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])

        with patch.object(era5, '_batch_get_from_cds') as mock_batch:
            era5.batch_fetch([(dt.datetime(2020, 1, 1, 0), missing)])

        mock_batch.assert_called_once()
        queued_paths = [p for _, p in mock_batch.call_args[0][0]]
        assert missing in queued_paths

    def test_mixed_existing_and_missing(self, era5: ERA5, tmp_path: Path) -> None:
        """Every path handed in is queued, whether or not it is already on disk."""
        existing = tmp_path / 'exists.nc'
        existing.touch()
        missing = tmp_path / 'new.nc'

        era5.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])

        with patch.object(era5, '_batch_get_from_cds') as mock_batch:
            era5.batch_fetch([
                (dt.datetime(2020, 1, 1, 0), existing),
                (dt.datetime(2020, 1, 2, 0), missing),
            ])

        mock_batch.assert_called_once()
        queued_paths = [p for _, p in mock_batch.call_args[0][0]]
        assert queued_paths == [existing, missing]

    def test_empty_input_does_not_call_batch(self, era5: ERA5, tmp_path: Path) -> None:
        era5.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])
        with patch.object(era5, '_batch_get_from_cds') as mock_batch:
            era5.batch_fetch([])
        mock_batch.assert_not_called()

    def test_time_rounding(self, era5: ERA5, tmp_path: Path) -> None:
        """Times that are not on an exact hour boundary must be rounded."""
        missing = tmp_path / 'rounded.nc'

        era5.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])

        with patch.object(era5, '_batch_get_from_cds') as mock_batch:
            era5.batch_fetch([(dt.datetime(2020, 1, 1, 0, 31), missing)])

        mock_batch.assert_called_once()
        queued_dt, _ = mock_batch.call_args[0][0][0]
        assert queued_dt.minute == 0


# ---------------------------------------------------------------------------
# ERA5T inherits ERA5 loading unchanged
# ---------------------------------------------------------------------------

class TestERA5TLoading:
    def test_load_weather_uses_pl_loader(self, era5t: ERA5T, tmp_path: Path) -> None:
        f = write_pl_file_batch_format(tmp_path / 'test.nc')
        era5t.setLevelType('pl')
        era5t.set_latlon_bounds([48.0, 53.0, 9.0, 15.0])

        era5t.load_weather(f)

        assert era5t._t is not None
        assert era5t._t.shape == (_NLAT, _NLON, _NLEV)

    def test_batch_fetch_inherited(self, era5t: ERA5T, tmp_path: Path) -> None:
        """ERA5T inherits batch_fetch from ERA5."""
        assert callable(getattr(era5t, 'batch_fetch', None))
