"""Tests for the multi-date ERA-5 download path and the level-type plumbing.

`ECMWF._batch_get_from_cds` is the point of the multiday-download work: it turns
N acquisition times into a *single* pair of CDS requests for model levels, then
splits the response into one raw file per datetime so that the rest of RAiDER
sees exactly what the old one-request-per-date path produced.

Nothing in CI performs a real CDS retrieval, so these tests stub out
`cdsapi.Client` and assert on how the download was split, on the per-datetime
files that come out the other side, and on the config plumbing that selects
model versus pressure levels.

The pressure-level path deliberately does *not* batch: MARS rejects more than
one date per month in a `reanalysis-era5-pressure-levels` request, so it issues
one request per datetime. That difference is asserted below so it does not
silently change.
"""

import argparse
import datetime as dt
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

import RAiDER.processWM
from RAiDER.cli.raider import collect_batch_times
from RAiDER.cli.validators import parse_weather_model
from RAiDER.models.era5 import ERA5
from RAiDER.models.era5t import ERA5T
from RAiDER.models.gmao import GMAO
from RAiDER.models.weatherModel import checkContainment_raw


G0 = 9.80665

NLEV_ML = 137
NLEV_PL = 8

LATS = np.arange(50.0, 48.9, -0.25)  # descending, as ERA-5 delivers them
LONS = np.arange(9.0, 10.1, 0.25)
NLAT, NLON = len(LATS), len(LONS)

BOUNDS = [49.0, 50.0, 9.0, 10.0]  # S, N, W, E -- comfortably inside the grid
# Strictly interior to the synthetic grid. shapely's `contains` excludes the
# boundary, so a containment check against BOUNDS itself would fail on the
# shared edges even though the cached file really does cover the request.
CONTAINED_BOUNDS = [49.2, 49.8, 9.2, 9.8]

SURFACE_PRESSURE = 95_000.0  # Pa
SURFACE_GEOPOTENTIAL = 500.0 * G0  # 500 m, in m^2/s^2

PL_LEVELS_HPA = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0, 50.0])


def _parse_cds_times(params: dict) -> list[np.datetime64]:
    """Expand a CDS request's date/time strings into their Cartesian product.

    CDS takes `date` and `time` as `/`-separated lists and returns every
    combination, which is exactly the batching trick `_batch_get_from_cds`
    relies on.
    """
    dates = params['date'].split('/')
    times = params['time'].split('/')
    return [np.datetime64(f'{d}T{t}:00', 'ns') for d in dates for t in times]


def _t_profile(nlev: int) -> np.ndarray:
    return np.linspace(220.0, 290.0, nlev)


def _write_ml_surface_file(path: Path, times: list[np.datetime64]) -> None:
    """Write the lnsp/z response: a single model level (level 1), N times."""
    nt = len(times)
    lnsp = np.full((nt, 1, NLAT, NLON), np.log(SURFACE_PRESSURE))
    z = np.full((nt, 1, NLAT, NLON), SURFACE_GEOPOTENTIAL)
    xr.Dataset(
        {
            'lnsp': (('valid_time', 'model_level', 'latitude', 'longitude'), lnsp),
            'z': (('valid_time', 'model_level', 'latitude', 'longitude'), z),
        },
        coords={
            'valid_time': np.array(times, dtype='datetime64[ns]'),
            'model_level': [1],
            'latitude': LATS,
            'longitude': LONS,
        },
    ).to_netcdf(path)


def _write_ml_file(path: Path, times: list[np.datetime64]) -> None:
    """Write the t/q response: all model levels, N times."""
    nt = len(times)
    t = np.broadcast_to(
        _t_profile(NLEV_ML)[np.newaxis, :, np.newaxis, np.newaxis],
        (nt, NLEV_ML, NLAT, NLON),
    ).copy()
    q = np.full((nt, NLEV_ML, NLAT, NLON), 1e-3)
    xr.Dataset(
        {
            't': (('valid_time', 'model_level', 'latitude', 'longitude'), t),
            'q': (('valid_time', 'model_level', 'latitude', 'longitude'), q),
        },
        coords={
            'valid_time': np.array(times, dtype='datetime64[ns]'),
            'model_level': np.arange(1, NLEV_ML + 1),
            'latitude': LATS,
            'longitude': LONS,
        },
    ).to_netcdf(path)


def _write_pl_file(path: Path, times: list[np.datetime64]) -> None:
    """Write a raw CDS pressure-level response: z as geopotential (m^2/s^2)."""
    nt = len(times)
    # Heights rising with decreasing pressure, in geopotential units
    z_ght = np.array([100.0, 1_500.0, 3_000.0, 5_500.0, 9_200.0, 11_800.0, 16_200.0, 20_600.0])
    z = np.broadcast_to(
        (z_ght * G0)[np.newaxis, :, np.newaxis, np.newaxis], (nt, NLEV_PL, NLAT, NLON)
    ).copy()
    t = np.broadcast_to(
        _t_profile(NLEV_PL)[::-1][np.newaxis, :, np.newaxis, np.newaxis],
        (nt, NLEV_PL, NLAT, NLON),
    ).copy()
    q = np.full((nt, NLEV_PL, NLAT, NLON), 1e-3)
    xr.Dataset(
        {
            'z': (('valid_time', 'pressure_level', 'latitude', 'longitude'), z),
            't': (('valid_time', 'pressure_level', 'latitude', 'longitude'), t),
            'q': (('valid_time', 'pressure_level', 'latitude', 'longitude'), q),
        },
        coords={
            'valid_time': np.array(times, dtype='datetime64[ns]'),
            'pressure_level': PL_LEVELS_HPA,
            'latitude': LATS,
            'longitude': LONS,
        },
    ).to_netcdf(path)


class FakeCDSClient:
    """Stand-in for `cdsapi.Client` that serves synthetic multi-time responses.

    Every request is recorded so tests can assert on how the download was split
    across datasets and datetimes.
    """

    url = 'https://cds.climate.copernicus.eu/api'
    requests: list[tuple[str, dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        FakeCDSClient.requests = []

    def retrieve(self, dataset: str, params: dict, target) -> None:
        FakeCDSClient.requests.append((dataset, dict(params)))
        times = _parse_cds_times(params)
        param = sorted(params['param'])

        if params.get('levtype') == 'pl':
            _write_pl_file(Path(target), times)
        elif param == ['lnsp', 'z']:
            _write_ml_surface_file(Path(target), times)
        elif param == ['q', 't']:
            _write_ml_file(Path(target), times)
        else:
            raise AssertionError(f'unexpected CDS request: {dataset} {param}')


@pytest.fixture()
def era5() -> ERA5:
    model = ERA5()
    model.setLevelType('ml')
    model.set_latlon_bounds(BOUNDS)
    return model


TIMES_3 = [
    dt.datetime(2020, 1, 1, 0),
    dt.datetime(2020, 1, 1, 12),
    dt.datetime(2020, 1, 2, 0),
]


@pytest.fixture()
def batched_ml(era5: ERA5, tmp_path: Path) -> list[tuple[dt.datetime, Path]]:
    """Run the model-level batch download for three datetimes across two days."""
    times_and_paths = [(t, tmp_path / f'ERA-5_{t:%Y%m%dT%H%M%S}.nc') for t in TIMES_3]
    with patch('cdsapi.Client', FakeCDSClient):
        era5._batch_get_from_cds(times_and_paths, *BOUNDS[:1], *BOUNDS[1:])
    return times_and_paths


# ---------------------------------------------------------------------------
# Model levels: the batching itself
# ---------------------------------------------------------------------------

class TestBatchModelLevels:
    def test_three_datetimes_cost_only_two_requests(self, batched_ml) -> None:
        """The whole point: N datetimes must not mean N (or 2N) CDS requests."""
        assert len(FakeCDSClient.requests) == 2

    def test_surface_and_model_fields_still_requested_separately(self, batched_ml) -> None:
        params = [sorted(p['param']) for _, p in FakeCDSClient.requests]
        assert params == [['lnsp', 'z'], ['q', 't']]

    def test_request_uses_era5_complete(self, batched_ml) -> None:
        assert {d for d, _ in FakeCDSClient.requests} == {'reanalysis-era5-complete'}

    def test_dates_are_deduplicated_in_the_request(self, batched_ml) -> None:
        """Two datetimes share 2020-01-01, which must appear once."""
        _, params = FakeCDSClient.requests[0]
        assert params['date'].split('/') == ['2020-01-01', '2020-01-02']

    def test_times_are_deduplicated_in_the_request(self, batched_ml) -> None:
        _, params = FakeCDSClient.requests[0]
        assert params['time'].split('/') == ['00:00', '12:00']

    def test_one_output_file_per_datetime(self, batched_ml) -> None:
        for _, path in batched_ml:
            assert path.exists(), f'{path} was not written'

    def test_each_file_holds_its_own_datetime(self, batched_ml) -> None:
        """The split must not smear one time's data across every output file."""
        for when, path in batched_ml:
            with xr.open_dataset(path) as ds:
                stored = ds['valid_time'].values
                assert stored.shape == (1,)
                assert stored[0] == np.datetime64(when, 'ns')

    def test_files_are_readable_by_makeDataCubes(self, era5: ERA5, batched_ml) -> None:
        """The batch writer and the reader have to agree on the layout."""
        for _, path in batched_ml:
            lats, lons, _, _, t, q, lnsp, z = era5._makeDataCubes(path)
            assert t.shape == (NLEV_ML, len(lats), len(lons))
            assert q.shape == t.shape
            assert z.shape == t.shape
            assert lnsp.shape == (len(lats), len(lons))

    def test_lnsp_round_trips_as_a_surface_field(self, era5: ERA5, batched_ml) -> None:
        _, _, _, _, _, _, lnsp, _ = era5._makeDataCubes(batched_ml[0][1])
        np.testing.assert_allclose(np.exp(lnsp), SURFACE_PRESSURE, rtol=1e-5)

    def test_z_is_a_full_model_level_cube(self, era5: ERA5, batched_ml) -> None:
        """CDS supplies z at the surface only; the batch writer fills every level."""
        _, _, _, _, _, _, _, z = era5._makeDataCubes(batched_ml[0][1])
        assert np.ptp(z[:, 0, 0]) > 1_000, 'z looks constant in level; calcgeoh did not run'

    def test_load_model_level_populates_the_model(self, era5: ERA5, batched_ml) -> None:
        era5._load_model_level(batched_ml[0][1])
        assert era5._t.shape == (NLAT, NLON, NLEV_ML)
        assert np.all(np.diff(era5._zs, axis=2) > 0), 'heights must increase surface -> TOA'
        assert era5._p[0, 0, 0] > era5._p[0, 0, -1], 'pressure must be surface-first'

    def test_single_datetime_goes_through_the_same_path(self, era5: ERA5, tmp_path: Path) -> None:
        """_get_from_cds is a one-element batch, so it must produce the same layout."""
        out = tmp_path / 'single.nc'
        with patch('cdsapi.Client', FakeCDSClient):
            era5._get_from_cds(*BOUNDS[:1], *BOUNDS[1:], dt.datetime(2020, 1, 1, 0), out)

        assert len(FakeCDSClient.requests) == 2
        _, _, _, _, t, _, lnsp, _ = era5._makeDataCubes(out)
        assert t.shape == (NLEV_ML, NLAT, NLON)
        assert lnsp.shape == (NLAT, NLON)

    def test_off_grid_time_is_rounded_before_the_request(self, era5: ERA5, tmp_path: Path) -> None:
        out = tmp_path / 'rounded.nc'
        with patch('cdsapi.Client', FakeCDSClient):
            era5._get_from_cds(*BOUNDS[:1], *BOUNDS[1:], dt.datetime(2020, 1, 1, 0, 31), out)

        _, params = FakeCDSClient.requests[0]
        assert params['time'] == '01:00'


# ---------------------------------------------------------------------------
# Pressure levels: deliberately one request per datetime
# ---------------------------------------------------------------------------

class TestBatchPressureLevels:
    @pytest.fixture()
    def batched_pl(self, era5: ERA5, tmp_path: Path):
        era5.setLevelType('pl')
        times_and_paths = [(t, tmp_path / f'pl_{t:%Y%m%dT%H%M%S}.nc') for t in TIMES_3]
        with patch('cdsapi.Client', FakeCDSClient):
            era5._batch_get_from_cds(times_and_paths, *BOUNDS[:1], *BOUNDS[1:])
        return times_and_paths

    def test_one_request_per_datetime(self, batched_pl) -> None:
        """MARS rejects >1 date per month here, so batching is not possible."""
        assert len(FakeCDSClient.requests) == len(TIMES_3)

    def test_uses_the_pressure_level_archive(self, batched_pl) -> None:
        assert {d for d, _ in FakeCDSClient.requests} == {'reanalysis-era5-pressure-levels'}

    def test_request_declares_product_type(self, batched_pl) -> None:
        """reanalysis-era5-pressure-levels rejects a request without product_type."""
        for _, params in FakeCDSClient.requests:
            assert params.get('product_type') == 'reanalysis'

    def test_each_request_asks_for_one_date(self, batched_pl) -> None:
        dates = [p['date'] for _, p in FakeCDSClient.requests]
        assert dates == [f'{t:%Y-%m-%d}' for t in TIMES_3]

    def test_one_output_file_per_datetime(self, batched_pl) -> None:
        for _, path in batched_pl:
            assert path.exists()

    def test_z_is_converted_to_height_and_labelled(self, batched_pl) -> None:
        """The writer divides by g0, and must record that so the reader knows."""
        with xr.open_dataset(batched_pl[0][1]) as ds:
            assert ds['z'].attrs.get('units') == 'm'
            assert ds['z'].values.max() < 50_000, 'z still looks like geopotential'

    def test_written_dims_are_lat_lon_level(self, batched_pl) -> None:
        with xr.open_dataset(batched_pl[0][1]) as ds:
            assert ds['t'].dims == ('valid_time', 'latitude', 'longitude', 'pressure_level')

    def test_files_round_trip_through_load_pressure_level(self, era5: ERA5, batched_pl) -> None:
        era5.setLevelType('pl')
        era5._load_pressure_level(batched_pl[0][1])

        assert era5._t.shape == (NLAT, NLON, NLEV_PL)
        assert np.all(np.diff(era5._zs, axis=2) > 0), 'heights must increase surface -> TOA'
        assert era5._p[0, 0, 0] > era5._p[0, 0, -1], 'pressure must be surface-first'

    def test_height_is_not_divided_by_g0_twice(self, era5: ERA5, batched_pl) -> None:
        """The writer already divided; the reader must not do it again."""
        era5.setLevelType('pl')
        era5._load_pressure_level(batched_pl[0][1])
        # Top level sits at ~20.6 km; a second division would put it near 2.1 km.
        assert era5._zs.max() > 15_000


class TestGeopotentialUnitsDetection:
    """_load_pressure_level has to tell geopotential from geopotential height."""

    def _write(self, path: Path, z_values: np.ndarray, units: str | None) -> Path:
        attrs = {'units': units} if units is not None else {}
        t = np.broadcast_to(
            _t_profile(len(z_values))[::-1][np.newaxis, np.newaxis, np.newaxis, :],
            (1, NLAT, NLON, len(z_values)),
        ).copy()
        xr.Dataset(
            {
                'z': xr.Variable(
                    ('valid_time', 'latitude', 'longitude', 'pressure_level'),
                    np.broadcast_to(
                        z_values[np.newaxis, np.newaxis, np.newaxis, :],
                        (1, NLAT, NLON, len(z_values)),
                    ).copy(),
                    attrs,
                ),
                't': xr.Variable(('valid_time', 'latitude', 'longitude', 'pressure_level'), t),
                'q': xr.Variable(
                    ('valid_time', 'latitude', 'longitude', 'pressure_level'),
                    np.full((1, NLAT, NLON, len(z_values)), 1e-3),
                ),
            },
            coords={
                'valid_time': np.array(['2020-01-01'], dtype='datetime64[ns]'),
                'latitude': LATS,
                'longitude': LONS,
                'pressure_level': PL_LEVELS_HPA[: len(z_values)],
            },
        ).to_netcdf(path)
        return path

    HEIGHTS = np.array([100.0, 1_500.0, 3_000.0, 5_500.0, 9_200.0, 11_800.0, 16_200.0, 20_600.0])

    def _top_height(self, era5: ERA5, path: Path) -> float:
        era5.setLevelType('pl')
        era5._load_pressure_level(path)
        return float(era5._zs.max())

    # `_get_heights` converts geopotential height to geometric height, which
    # moves the top of this column by ~0.3%. Tolerances below are loose enough
    # to ignore that but far tighter than a stray factor of g0 (~10x).
    RTOL = 1e-2

    def test_units_attribute_marks_geopotential(self, era5: ERA5, tmp_path: Path) -> None:
        f = self._write(tmp_path / 'gp.nc', self.HEIGHTS * G0, 'm**2 s**-2')
        np.testing.assert_allclose(self._top_height(era5, f), self.HEIGHTS.max(), rtol=self.RTOL)

    def test_units_attribute_marks_height(self, era5: ERA5, tmp_path: Path) -> None:
        f = self._write(tmp_path / 'ght.nc', self.HEIGHTS, 'm')
        np.testing.assert_allclose(self._top_height(era5, f), self.HEIGHTS.max(), rtol=self.RTOL)

    def test_both_unit_conventions_agree(self, tmp_path: Path) -> None:
        """The same column written either way must load to the same heights."""
        gp = self._write(tmp_path / 'gp2.nc', self.HEIGHTS * G0, 'm**2 s**-2')
        ght = self._write(tmp_path / 'ght2.nc', self.HEIGHTS, 'm')

        heights = []
        for f in (gp, ght):
            model = ERA5()
            model.set_latlon_bounds(BOUNDS)
            heights.append(self._top_height(model, f))

        np.testing.assert_allclose(heights[0], heights[1], rtol=1e-6)

    def test_units_win_over_magnitude_for_a_shallow_column(self, era5: ERA5, tmp_path: Path) -> None:
        """A geopotential file topping out below ~10 km is the case the old
        magnitude-only heuristic got wrong: max(z) is under 100,000 m**2 s**-2,
        so it would have skipped the g0 division and reported heights ~10x too big.
        """
        shallow = np.array([100.0, 1_500.0, 3_000.0, 5_500.0])
        f = self._write(tmp_path / 'shallow.nc', shallow * G0, 'm**2 s**-2')
        assert (shallow * G0).max() < 100_000, 'test no longer probes the ambiguous range'

        np.testing.assert_allclose(self._top_height(era5, f), shallow.max(), rtol=self.RTOL)

    def test_magnitude_fallback_when_units_are_absent(self, era5: ERA5, tmp_path: Path) -> None:
        f = self._write(tmp_path / 'nounits.nc', self.HEIGHTS * G0, None)
        np.testing.assert_allclose(self._top_height(era5, f), self.HEIGHTS.max(), rtol=self.RTOL)


# ---------------------------------------------------------------------------
# processWM.batch_download_weather_model
# ---------------------------------------------------------------------------

class TestBatchDownloadWeatherModel:
    def test_queues_every_missing_time(self, era5: ERA5, tmp_path: Path) -> None:
        era5.set_wmLoc(str(tmp_path))
        with patch.object(era5, 'batch_fetch') as fetch:
            RAiDER.processWM.batch_download_weather_model(era5, TIMES_3, BOUNDS)

        fetch.assert_called_once()
        queued = fetch.call_args[0][0]
        assert [t for t, _ in queued] == TIMES_3

    def test_writes_into_the_weather_model_directory(self, era5: ERA5, tmp_path: Path) -> None:
        wm_dir = tmp_path / 'weather_files'
        era5.set_wmLoc(str(wm_dir))
        with patch.object(era5, 'batch_fetch') as fetch:
            RAiDER.processWM.batch_download_weather_model(era5, TIMES_3[:1], BOUNDS)

        queued_path = fetch.call_args[0][0][0][1]
        assert queued_path.parent == wm_dir
        assert wm_dir.is_dir(), 'the destination directory must be created up front'

    def test_no_call_when_nothing_is_missing(self, era5: ERA5, tmp_path: Path) -> None:
        """An existing file that covers the bounds must not be re-downloaded."""
        era5.set_wmLoc(str(tmp_path))
        for t in TIMES_3:
            path = Path(
                RAiDER.processWM.make_raw_weather_data_filename(str(tmp_path), era5.Model(), t)
            )
            _write_pl_file(path, [np.datetime64(t, 'ns')])

        with patch.object(era5, 'batch_fetch') as fetch:
            RAiDER.processWM.batch_download_weather_model(era5, TIMES_3, CONTAINED_BOUNDS)

        fetch.assert_not_called()

    def test_force_download_requeues_existing_files(self, era5: ERA5, tmp_path: Path) -> None:
        era5.set_wmLoc(str(tmp_path))
        for t in TIMES_3:
            path = Path(
                RAiDER.processWM.make_raw_weather_data_filename(str(tmp_path), era5.Model(), t)
            )
            _write_pl_file(path, [np.datetime64(t, 'ns')])

        with patch.object(era5, 'batch_fetch') as fetch:
            RAiDER.processWM.batch_download_weather_model(
                era5, TIMES_3, CONTAINED_BOUNDS, force_download=True
            )

        fetch.assert_called_once()
        assert len(fetch.call_args[0][0]) == len(TIMES_3)

    def test_existing_file_that_is_too_small_is_requeued(self, era5: ERA5, tmp_path: Path) -> None:
        """A cached file that does not cover the requested bounds is not reusable."""
        era5.set_wmLoc(str(tmp_path))
        t = TIMES_3[0]
        path = Path(
            RAiDER.processWM.make_raw_weather_data_filename(str(tmp_path), era5.Model(), t)
        )
        _write_pl_file(path, [np.datetime64(t, 'ns')])

        # Far wider than the synthetic grid, so containment fails
        with patch.object(era5, 'batch_fetch') as fetch:
            RAiDER.processWM.batch_download_weather_model(era5, [t], [0.0, 80.0, -170.0, 170.0])

        fetch.assert_called_once()

    def test_empty_time_list_is_a_no_op(self, era5: ERA5, tmp_path: Path) -> None:
        era5.set_wmLoc(str(tmp_path))
        with patch.object(era5, 'batch_fetch') as fetch:
            RAiDER.processWM.batch_download_weather_model(era5, [], BOUNDS)
        fetch.assert_not_called()

    def test_datetime_outside_the_valid_range_is_skipped(self, era5: ERA5, tmp_path: Path) -> None:
        """checkTime screens the batch, as fetch() screens the per-date path.

        ERA-5 lags real time by three months. Sending a too-recent datetime to CDS
        would fail the whole request and, with it, every good date in the stack.
        """
        era5.set_wmLoc(str(tmp_path))
        too_recent = dt.datetime.now() - dt.timedelta(days=5)

        with patch.object(era5, 'batch_fetch') as fetch:
            RAiDER.processWM.batch_download_weather_model(era5, [TIMES_3[0], too_recent], BOUNDS)

        queued = [t for t, _ in fetch.call_args[0][0]]
        assert queued == [TIMES_3[0]], 'the out-of-range datetime must not reach CDS'

    def test_every_datetime_out_of_range_means_no_request(self, era5: ERA5, tmp_path: Path) -> None:
        era5.set_wmLoc(str(tmp_path))
        too_early = dt.datetime(1949, 12, 31, 0)

        with patch.object(era5, 'batch_fetch') as fetch:
            RAiDER.processWM.batch_download_weather_model(era5, [too_early], BOUNDS)

        fetch.assert_not_called()

    def test_force_download_actually_downloads(self, era5: ERA5, tmp_path: Path) -> None:
        """force_download must reach CDS, not just reach batch_fetch.

        batch_fetch used to re-filter on out_path.exists(), so the queue the caller
        built was discarded and force_download downloaded nothing. Mocking batch_fetch
        hides that, so this test mocks only the CDS client.
        """
        era5.set_wmLoc(str(tmp_path))
        for t in TIMES_3:
            path = Path(
                RAiDER.processWM.make_raw_weather_data_filename(str(tmp_path), era5.Model(), t)
            )
            _write_pl_file(path, [np.datetime64(t, 'ns')])

        with patch('cdsapi.Client', FakeCDSClient):
            RAiDER.processWM.batch_download_weather_model(
                era5, TIMES_3, CONTAINED_BOUNDS, force_download=True
            )

        assert FakeCDSClient.requests, 'force_download performed no download at all'

    def test_too_small_cached_file_is_actually_redownloaded(self, era5: ERA5, tmp_path: Path) -> None:
        """A cached file that fails containment must be refreshed, not dropped."""
        era5.set_wmLoc(str(tmp_path))
        t = TIMES_3[0]
        path = Path(
            RAiDER.processWM.make_raw_weather_data_filename(str(tmp_path), era5.Model(), t)
        )
        _write_pl_file(path, [np.datetime64(t, 'ns')])

        with patch('cdsapi.Client', FakeCDSClient):
            RAiDER.processWM.batch_download_weather_model(era5, [t], [0.0, 80.0, -170.0, 170.0])

        assert FakeCDSClient.requests, 'the undersized cached file was never re-fetched'

    def test_existing_processed_file_is_not_redownloaded(self, era5: ERA5, tmp_path: Path) -> None:
        """prepareWeatherModel accepts a processed file with no raw file present.

        Deleting the raw files while keeping the processed ones is the standard
        disk-saving pattern here, so the batch must not re-request data that
        prepareWeatherModel is about to ignore.
        """
        era5.set_wmLoc(str(tmp_path))
        t = TIMES_3[0]
        era5.setTime(t)
        Path(era5.out_file(str(tmp_path))).parent.mkdir(parents=True, exist_ok=True)
        Path(era5.out_file(str(tmp_path))).touch()

        with patch.object(era5, 'batch_fetch') as fetch:
            RAiDER.processWM.batch_download_weather_model(era5, [t], BOUNDS)

        fetch.assert_not_called()

    def test_force_download_overrides_the_processed_file(self, era5: ERA5, tmp_path: Path) -> None:
        era5.set_wmLoc(str(tmp_path))
        t = TIMES_3[0]
        era5.setTime(t)
        Path(era5.out_file(str(tmp_path))).parent.mkdir(parents=True, exist_ok=True)
        Path(era5.out_file(str(tmp_path))).touch()

        with patch.object(era5, 'batch_fetch') as fetch:
            RAiDER.processWM.batch_download_weather_model(era5, [t], BOUNDS, force_download=True)

        fetch.assert_called_once()


# ---------------------------------------------------------------------------
# checkContainment_raw
# ---------------------------------------------------------------------------

class TestCheckContainmentRaw:
    """Does a cached raw file cover the requested bounds?

    The previous implementation answered `False` for every weather model that
    sat inside the world box -- i.e. every regional model -- because the final
    branch returned a bare `False` instead of testing the input box. That made
    every cached raw file look unusable, so `prepareWeatherModel` re-downloaded
    data it already had.
    """

    @pytest.fixture()
    def cached(self, tmp_path: Path) -> Path:
        path = tmp_path / 'cached.nc'
        _write_pl_file(path, [np.datetime64('2020-01-01', 'ns')])
        return path

    def test_interior_request_is_contained(self, cached: Path) -> None:
        assert checkContainment_raw(cached, CONTAINED_BOUNDS) is True

    def test_request_larger_than_the_model_is_not_contained(self, cached: Path) -> None:
        assert checkContainment_raw(cached, [0.0, 80.0, -170.0, 170.0]) is False

    def test_partially_overlapping_request_is_not_contained(self, cached: Path) -> None:
        # Slides east past the model's 10 deg edge
        assert checkContainment_raw(cached, [49.2, 49.8, 9.5, 12.0]) is False

    def test_disjoint_request_is_not_contained(self, cached: Path) -> None:
        assert checkContainment_raw(cached, [20.0, 21.0, 100.0, 101.0]) is False


# ---------------------------------------------------------------------------
# collect_batch_times: which datetimes the batch has to cover
# ---------------------------------------------------------------------------

class TestCollectBatchTimes:
    T0 = dt.datetime(2020, 1, 1, 12, 0)

    def test_no_interpolation_keeps_the_acquisition_times(self) -> None:
        assert collect_batch_times([self.T0], None, 6) == [self.T0]

    def test_none_string_is_treated_as_no_interpolation(self) -> None:
        assert collect_batch_times([self.T0], 'none', 6) == [self.T0]

    def test_center_time_pulls_in_the_bracketing_model_times(self) -> None:
        out = collect_batch_times([dt.datetime(2020, 1, 1, 13, 0)], 'center_time', 6)
        assert out == [dt.datetime(2020, 1, 1, 12, 0), dt.datetime(2020, 1, 1, 18, 0)]

    def test_azimuth_time_grid_pulls_in_a_wider_window(self) -> None:
        out = collect_batch_times([dt.datetime(2020, 1, 1, 13, 0)], 'azimuth_time_grid', 6)
        assert len(out) >= 2
        assert all(isinstance(t, dt.datetime) for t in out)

    def test_overlapping_windows_are_deduplicated(self) -> None:
        """Two acquisitions an hour apart share their bracketing model times."""
        out = collect_batch_times(
            [dt.datetime(2020, 1, 1, 13, 0), dt.datetime(2020, 1, 1, 14, 0)],
            'center_time',
            6,
        )
        assert len(out) == len(set(out)) == 2

    def test_order_is_preserved(self) -> None:
        times = [dt.datetime(2020, 1, 3), dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)]
        assert collect_batch_times(times, None, 6) == times

    def test_missing_model_step_falls_back_to_six_hours(self) -> None:
        out = collect_batch_times([dt.datetime(2020, 1, 1, 13, 0)], 'center_time', None)
        assert out == [dt.datetime(2020, 1, 1, 12, 0), dt.datetime(2020, 1, 1, 18, 0)]


# ---------------------------------------------------------------------------
# Level-type selection: setLevelType, config key, GUNW flag
# ---------------------------------------------------------------------------

class TestSetLevelType:
    @pytest.mark.parametrize(
        ('given', 'expected'),
        [('ml', 'ml'), ('pl', 'pl'), ('model', 'ml'), ('pressure', 'pl'),
         ('MODEL', 'ml'), ('Pressure', 'pl')],
    )
    def test_aliases_map_onto_internal_codes(self, era5: ERA5, given: str, expected: str) -> None:
        era5.setLevelType(given)
        assert era5._model_level_type == expected

    def test_unknown_level_type_is_rejected(self, era5: ERA5) -> None:
        with pytest.raises(RuntimeError):
            era5.setLevelType('isentropic')

    def test_level_type_drives_the_dataset_choice(self, era5: ERA5, tmp_path: Path) -> None:
        era5.setLevelType('pressure')
        with patch('cdsapi.Client', FakeCDSClient):
            era5._batch_get_from_cds(
                [(TIMES_3[0], tmp_path / 'x.nc')], *BOUNDS[:1], *BOUNDS[1:]
            )
        assert FakeCDSClient.requests[0][0] == 'reanalysis-era5-pressure-levels'


class TestParseWeatherModelLevelType:
    class _AOI:
        def bounds(self):
            return BOUNDS

    def test_default_is_left_alone(self) -> None:
        model = parse_weather_model('ERA5', self._AOI())
        assert model._model_level_type == 'ml'

    @pytest.mark.parametrize(('given', 'expected'), [('pl', 'pl'), ('pressure', 'pl'), ('model', 'ml')])
    def test_level_type_is_applied(self, given: str, expected: str) -> None:
        model = parse_weather_model('ERA5', self._AOI(), level_type=given)
        assert model._model_level_type == expected

    def test_invalid_level_type_names_the_model(self) -> None:
        with pytest.raises(ValueError, match='ERA5'):
            parse_weather_model('ERA5', self._AOI(), level_type='isentropic')


class TestModelsWithoutPressureLevels:
    """GMAO/MERRA-2/NCMR only publish model levels; asking for pressure must fail."""

    def test_gmao_declares_model_level_support(self) -> None:
        gmao = GMAO()
        gmao.__model_levels__()
        assert gmao._zlevels is not None

    def test_gmao_rejects_pressure_levels(self) -> None:
        with pytest.raises(NotImplementedError, match='pressure levels'):
            GMAO().__pressure_levels__()

    def test_the_refusal_names_the_model(self) -> None:
        with pytest.raises(NotImplementedError, match='GMAO'):
            GMAO().__pressure_levels__()


class TestCalcDelaysGUNWModelLevels:
    """The --model-levels flag has to reach the generated run config."""

    def test_invalid_choice_is_rejected(self) -> None:
        from RAiDER.cli.raider import calcDelaysGUNW

        # argparse rejects this before any file or network access
        with pytest.raises(SystemExit):
            calcDelaysGUNW(['-f', 'unused.nc', '-m', 'ERA5', '--model-levels', 'isentropic'])

    @pytest.mark.parametrize('level', ['ml', 'model', 'pl', 'pressure'])
    def test_level_reaches_the_run_config(self, level: str, tmp_path: Path) -> None:
        import RAiDER.aria.prepFromGUNW as prep

        args = argparse.Namespace(
            weather_model='ERA5',
            model_levels=level,
            api_uid=None,
            api_key=None,
            file=Path('unused.nc'),
            output_directory=tmp_path,
            interpolate_time='none',
        )

        written = {}
        with patch.object(prep.credentials, 'check_api'), \
             patch.object(prep, 'GUNW', _FakeGUNW), \
             patch.object(prep, 'write_yaml', lambda cfg, path: written.update(cfg)), \
             patch('RAiDER.aria.prepFromGUNW.Path', Path):
            prep.main(args)

        assert written['weather_model_levels'] == level

    def test_key_is_omitted_when_unset(self, tmp_path: Path) -> None:
        """No flag means the model's built-in default must stand."""
        import RAiDER.aria.prepFromGUNW as prep

        args = argparse.Namespace(
            weather_model='ERA5',
            model_levels=None,
            api_uid=None,
            api_key=None,
            file=Path('unused.nc'),
            output_directory=tmp_path,
            interpolate_time='none',
        )

        written = {}
        with patch.object(prep.credentials, 'check_api'), \
             patch.object(prep, 'GUNW', _FakeGUNW), \
             patch.object(prep, 'write_yaml', lambda cfg, path: written.update(cfg)):
            prep.main(args)

        assert 'weather_model_levels' not in written


class _FakeGUNW:
    """Minimal stand-in for the GUNW reader, which otherwise needs a real product."""

    def __init__(self, *args, **kwargs) -> None:
        self.look_dir = 'right'
        self.SNWE = BOUNDS
        self.heights = [0, 500]
        self.dates = [dt.datetime(2020, 1, 1)]
        self.mid_time = '12:00:00'
        self.orbit_file = 'orbit.EOF'
        self.spacing_m = 1000
        self.name = 'fake'
        self.wavelength = 0.055


class TestERA5TInheritsBatching:
    def test_era5t_batches_like_era5(self, tmp_path: Path) -> None:
        model = ERA5T()
        model.setLevelType('ml')
        model.set_latlon_bounds(BOUNDS)
        times_and_paths = [(t, tmp_path / f'{t:%Y%m%dT%H%M%S}.nc') for t in TIMES_3]

        with patch('cdsapi.Client', FakeCDSClient):
            model._batch_get_from_cds(times_and_paths, *BOUNDS[:1], *BOUNDS[1:])

        assert len(FakeCDSClient.requests) == 2
        for _, path in times_and_paths:
            assert path.exists()


# ---------------------------------------------------------------------------
# calcDelays: a partly-failed multi-date run must not exit successfully
# ---------------------------------------------------------------------------

class TestCalcDelaysFailedTimes:
    """`failed_times` has to count every way a date can be abandoned.

    The loop abandons a date at three separate `continue`s: no weather files, no
    weather-model file from `getWeatherFile`, and a `RuntimeError` out of
    `tropo_delay`. If only the first is recorded, a run that fails a different way at
    each date never reaches `len(failed_times) == len(date_list)`, so it returns an
    empty list and exits 0 having written nothing -- the worst outcome for a batch
    pipeline, since downstream automation sees success and an empty directory.
    """

    @staticmethod
    def _run_config(tmp_path: Path) -> Path:
        from RAiDER.utilFcns import write_yaml

        grp = {
            'date_group': {'date_start': '20200101', 'date_end': '20200102'},
            'time_group': {'time': '00:00:00', 'interpolate_time': 'none'},
            'weather_model': 'ERA5',
            'aoi_group': {'bounding_box': [49.0, 50.0, 9.0, 10.0]},
            'runtime_group': {
                'output_directory': str(tmp_path / 'output'),
                'weather_model_directory': str(tmp_path / 'weather_files'),
            },
            'verbose': False,
        }
        return Path(write_yaml(grp, tmp_path / 'run_config.yaml'))

    def test_mixed_failures_across_two_dates_still_raise(self, tmp_path: Path) -> None:
        import RAiDER.delay
        from RAiDER.cli.raider import calcDelays
        from RAiDER.models.customExceptions import NoWeatherModelData

        cfg = self._run_config(tmp_path)
        wm_file = tmp_path / 'wm.nc'
        wm_file.touch()

        calls = {'n': 0}

        def fake_prepare(model, tt, bounds, **kwargs):
            # First date: nothing usable. Second date: a file, which then fails later.
            calls['n'] += 1
            if calls['n'] == 1:
                raise ValueError('synthetic download failure')
            return str(wm_file)

        def fake_tropo_delay(*args, **kwargs):
            raise RuntimeError('synthetic delay failure')

        with patch.object(RAiDER.processWM, 'batch_download_weather_model'), \
             patch.object(RAiDER.processWM, 'prepareWeatherModel', fake_prepare), \
             patch('RAiDER.cli.raider.getWeatherFile', return_value=wm_file), \
             patch.object(RAiDER.delay, 'tropo_delay', fake_tropo_delay):
            with pytest.raises(NoWeatherModelData):
                calcDelays([str(cfg)])

    def test_a_run_that_fully_succeeds_does_not_raise(self, tmp_path: Path) -> None:
        """The guard must not fire when every date produced output."""
        import RAiDER.delay
        from RAiDER.cli.raider import calcDelays

        cfg = self._run_config(tmp_path)
        wm_file = tmp_path / 'wm.nc'
        wm_file.touch()

        def fake_tropo_delay(*args, **kwargs):
            # hydro_delay is not None, so the station/raster branch is taken
            return xr.Dataset(), xr.Dataset()

        with patch.object(RAiDER.processWM, 'batch_download_weather_model'), \
             patch.object(RAiDER.processWM, 'prepareWeatherModel', return_value=str(wm_file)), \
             patch('RAiDER.cli.raider.getWeatherFile', return_value=wm_file), \
             patch.object(RAiDER.delay, 'tropo_delay', fake_tropo_delay):
            out = calcDelays([str(cfg)])

        assert len(out) == 2
