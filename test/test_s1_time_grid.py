import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import RAiDER.s1_azimuth_timing
from RAiDER.s1_azimuth_timing import (
    get_inverse_weights_for_dates, get_n_closest_datetimes,
    get_s1_azimuth_time_grid, get_slc_id_from_point_and_time,
    get_times_for_azimuth_interpolation
)


def get_start_time_from_slc_id(slc_id: str) -> datetime.datetime:
    acq_start_time_token = slc_id.split('_')[5]
    return pd.Timestamp(acq_start_time_token)


def test_get_slc_id():
    """
    Function tested gets SLC with respect to space and time.

    Test is derived using a grid similar to:
    S1-GUNW-A-R-064-tops-20210723_20210711-015000-00119W_00033N-PP-6267-v2_0_6.nc
    Over Los Angeles.

    Makes sure that there is no SLC 10 degrees translated along longitude at exactly same time.
    """

    lon = np.linspace(-119, -114, 6)
    lat = np.linspace(36, 33, 4)

    lat_center = np.mean(lat)
    lon_center = np.mean(lon)

    ref_time = datetime.datetime(2021, 7, 23, 1, 50, 0)
    ref_slc_ids = ['S1B_IW_SLC__1SDV_20210723T014947_20210723T015014_027915_0354B4_B3A9']

    sec_time = datetime.datetime(2021, 7, 11, 1, 50, 0)
    sec_slc_ids = ['S1B_IW_SLC__1SDV_20210711T014922_20210711T014949_027740_034F80_859D',
                   'S1B_IW_SLC__1SDV_20210711T014947_20210711T015013_027740_034F80_D404',
                   'S1B_IW_SLC__1SDV_20210711T015011_20210711T015038_027740_034F80_376C']

    ids = [ref_slc_ids, sec_slc_ids]
    times = [ref_time, sec_time]
    for time, slc_ids_expected in zip(times, ids):
        slc_ids_retrieved = get_slc_id_from_point_and_time(lon_center, lat_center, time)
        assert [slc_id in slc_ids_expected for slc_id in slc_ids_retrieved]

    for time, slc_ids in zip(times, ids):
        with pytest.raises(ValueError):
            _ = get_slc_id_from_point_and_time(lon_center + 10, lat_center, time)


@pytest.mark.parametrize('ifg_type', ['reference', 'secondary'],)
def test_s1_timing_array_wrt_slc_center_time(gunw_azimuth_test: Path,
                                             ifg_type: str,
                                             orbit_dict_for_azimuth_time_test: dict,
                                             slc_id_dict_for_azimuth_time_test: dict,
                                             mocker):
    """Make sure the SLC start time is within reasonable amount of grid. The flow chart is:

    (datetime, lon, lat) --> SLC id --> orbit --> azimuth time grid (via ISCE3)

    The input (leftmost) datetime should not deviate too much from azimuth time grid and that is the content of test.
    """

    group = 'science/grids/imagingGeometry'
    with xr.open_dataset(gunw_azimuth_test, group=group, mode='r') as ds:
        res_x, res_y = ds.rio.resolution()
        # assuming ul corner centered
        lat = ds.latitudeMeta.data - res_y / 2.
        lon = ds.longitudeMeta.data - res_x / 2.
        hgt = ds.heightsMeta.data
    group = 'science/radarMetaData/inputSLC'
    with xr.open_dataset(gunw_azimuth_test, group=f'{group}/{ifg_type}') as ds:
        slc_ids = ds['L1InputGranules'].data
        # Ensure non-empty and sorted by acq_time
        slc_ids = sorted(list(filter(lambda x: x, slc_ids)))

    # Get the middle SLC start_time
    n = len(slc_ids)
    slc_start_time = get_start_time_from_slc_id(slc_ids[n // 2]).to_pydatetime()

    # Azimuth time grid
    mocker.patch(
        'RAiDER.s1_azimuth_timing.get_orbits_from_slc_ids_hyp3lib',
        side_effect=[
            [Path(orbit_dict_for_azimuth_time_test[ifg_type])],
        ]
    )

    mocker.patch('RAiDER.s1_azimuth_timing._asf_query',
                 return_value=slc_id_dict_for_azimuth_time_test[ifg_type])
    time_grid = get_s1_azimuth_time_grid(lon, lat, hgt, slc_start_time)

    abs_diff = np.abs(time_grid - np.datetime64(slc_start_time)) / np.timedelta64(1, 's')
    # Assert the absolute difference is less than 40 seconds from start time
    # Recall the SLC image spans approximately 30 seconds, but this grid is 1/3-1/2 bigger on each end.
    # And our time we are comparing against is a *start_time*
    assert np.all(abs_diff < 40)

    assert RAiDER.s1_azimuth_timing._asf_query.call_count == 1
    assert RAiDER.s1_azimuth_timing.get_orbits_from_slc_ids_hyp3lib.call_count == 1


@pytest.mark.parametrize('ifg_type', ['reference', 'secondary'])
def test_s1_timing_array_wrt_variance(gunw_azimuth_test: Path,
                                      ifg_type: str,
                                      orbit_dict_for_azimuth_time_test: dict,
                                      slc_id_dict_for_azimuth_time_test: dict,
                                      mocker):
    """Make sure along the hgt dimension of grid there is very small deviations
    """
    group = 'science/grids/imagingGeometry'
    with xr.open_dataset(gunw_azimuth_test, group=group, mode='r') as ds:
        res_x, res_y = ds.rio.resolution()
        # assuming ul corner centered
        lat = ds.latitudeMeta.data - res_y / 2.
        lon = ds.longitudeMeta.data - res_x / 2.
        hgt = ds.heightsMeta.data

    group = 'science/radarMetaData/inputSLC'
    with xr.open_dataset(gunw_azimuth_test, group=f'{group}/{ifg_type}') as ds:
        slc_ids = ds['L1InputGranules'].data
        # Ensure non-empty and sorted by acq_time
        slc_ids = sorted(list(filter(lambda x: x, slc_ids)))

    # Get the middle SLC start_time
    slc_start_time = get_start_time_from_slc_id(slc_ids[0]).to_pydatetime()

    # Azimuth time grid
    mocker.patch(
        'RAiDER.s1_azimuth_timing.get_orbits_from_slc_ids_hyp3lib',
        side_effect=[
            [Path(orbit_dict_for_azimuth_time_test[ifg_type])],
        ]
    )

    mocker.patch('RAiDER.s1_azimuth_timing._asf_query',
                 return_value=slc_id_dict_for_azimuth_time_test[ifg_type])
    X = get_s1_azimuth_time_grid(lon, lat, hgt, slc_start_time)

    Z = (X - X.min()) / np.timedelta64(1, 's')
    std_hgt = Z.std(axis=0).max()
    # Asserts the standard deviation in height is less than 2e-3 seconds
    assert np.all(std_hgt < 2e-3)

    assert RAiDER.s1_azimuth_timing._asf_query.call_count == 1
    assert RAiDER.s1_azimuth_timing.get_orbits_from_slc_ids_hyp3lib.call_count == 1


def test_n_closest_dts():
    """Check that the n closest datetimes are correct and in correct order.
    Order being absolute distance to supplied datetime"""
    n_target_datetimes = 3
    time_step = 6
    dt = datetime.datetime(2023, 1, 1, 11, 1, 1)
    out = get_n_closest_datetimes(dt, n_target_datetimes, time_step)
    expected = [datetime.datetime(2023, 1, 1, 12, 0, 0),
                datetime.datetime(2023, 1, 1, 6, 0, 0),
                datetime.datetime(2023, 1, 1, 18, 0, 0)]
    assert out == expected

    n_target_datetimes = 4
    time_step = 2
    dt = datetime.datetime(2023, 2, 1, 8, 1, 1)
    out = get_n_closest_datetimes(dt, n_target_datetimes, time_step)
    expected = [datetime.datetime(2023, 2, 1, 8, 0, 0),
                datetime.datetime(2023, 2, 1, 10, 0, 0),
                datetime.datetime(2023, 2, 1, 6, 0, 0),
                datetime.datetime(2023, 2, 1, 12, 0, 0)]
    assert out == expected

    n_target_datetimes = 2
    time_step = 4
    dt = datetime.datetime(2023, 1, 1, 20, 1, 1)
    out = get_n_closest_datetimes(dt, n_target_datetimes, time_step)
    expected = [datetime.datetime(2023, 1, 1, 20, 0, 0),
                datetime.datetime(2023, 1, 2, 0, 0, 0)]
    assert out == expected

    # Make sure if ref_time occurs at model time then we get correct times
    n_target_datetimes = 3
    time_step = 1
    dt = datetime.datetime(2023, 1, 2, 0, 0, 0)
    out = get_n_closest_datetimes(dt, n_target_datetimes, time_step)
    expected = [datetime.datetime(2023, 1, 2, 0, 0, 0),
                # Note we order equal distance from ref time by how early it is
                datetime.datetime(2023, 1, 1, 23, 0, 0),
                datetime.datetime(2023, 1, 2, 1, 0, 0),
               ]

    assert out == expected

    n_target_datetimes = 2
    # Does not divide 24 hours so has period > 1 day.
    time_step = 5
    dt = datetime.datetime(2023, 1, 1, 20, 1, 1)
    with pytest.raises(ValueError):
        _ = get_n_closest_datetimes(dt, n_target_datetimes, time_step)


input_times = [np.datetime64('2021-01-01T07:00:00'),
               np.datetime64('2021-01-01T07:00:00'),
               np.datetime64('2021-01-01T06:00:00')
               ]
# Windows are in hours
windows = [6, 3, 6]
expected_weights_list = [[.833, .167, 0],
                         [1., 0., 0.],
                         [1., 0., 0.]]


@pytest.mark.parametrize('input_time, temporal_window, expected_weights',
                         zip(input_times, windows, expected_weights_list))
def test_inverse_weighting(input_time: np.datetime64,
                           temporal_window: Union[int, float],
                           expected_weights: list[float]):
    """The test is designed to determine valid inverse weighting

    Parameters
    ----------
    input_time : np.datetime64
        This input datetime is used to create a 4 x 4  array X such that
        X[0, 0] = input time and the remainder are incremented by 1 second
    temporal_window : int | float
        The size (in hours) to consider for weighting
    expected_weights : float
        This is a single float because the 1 second delta does (in X construction)
        does not impact weights significantly (only tested 1e-3)
    """

    N = 4

    date_0 = np.datetime64('2021-01-01T06:00:00')
    date_1 = np.datetime64('2021-01-01T12:00:00')
    date_2 = np.datetime64('2021-01-01T00:00:00')
    dates = [date_0, date_1, date_2]
    dates = list(map(lambda dt: dt.astype(datetime.datetime), dates))

    timing_grid = np.full((N, N),
                          input_time,
                          dtype='datetime64[ms]')
    delta = np.timedelta64(1, 's') * np.arange(N**2)
    delta = delta.reshape((N, N))
    timing_grid += delta

    out_weights = get_inverse_weights_for_dates(timing_grid,
                                                dates,
                                                temporal_window_hours=temporal_window
                                                )

    # Note the delta makes all entries outside of top left corner slightly different
    # Due to 1 second delta added - however since model time step is in hours,
    # The weights are fairly close across 4 x 4 array
    for k in range(3):
        np.testing.assert_almost_equal(expected_weights[k],
                                       out_weights[k],
                                       1e-3)

    # a retrieval r_i on date_i should be interpolated as
    # w_0 * r_0 + ... + r_n * w_n and so we expect w's to have a sum of 1s
    # per pixel
    sum_weights_per_pixel = np.stack(out_weights, axis=1).sum(axis=1)
    np.testing.assert_almost_equal(1, sum_weights_per_pixel)


def test_triple_date_usage():
    """This test shows that when a time grid has pixels that are closer to two different pairs of date and that the
    inverse weights come out correctly. Put slightly differently, the scenario is when one of the 3 closest dates
    occurs at "center time" of the grid and some pixels require dates before the center time and some pixels
    require dates after center time.
    """
    date_0 = np.datetime64('2021-01-01T00:00:00')
    date_1 = np.datetime64('2021-01-01T06:00:00')
    date_2 = np.datetime64('2020-12-31T18:00:00')

    dates = [date_0, date_1, date_2]
    dates = list(map(lambda dt: dt.astype(datetime.datetime), dates))

    timing_grid = np.full((3,),
                          np.datetime64('2021-01-01T00:00:00'),
                          dtype='datetime64[ms]')
    delta = np.timedelta64(1, 's') * np.array([-10_000, 0, 10_000])
    timing_grid += delta

    out_weights = get_inverse_weights_for_dates(timing_grid,
                                                dates,
                                                temporal_window_hours=6,
                                                inverse_regularizer=1e-10
                                                )

    w_0 = out_weights[0]  # for date_0
    w_1 = out_weights[1]  # for date_1
    w_2 = out_weights[2]  # for date_2

    assert all([w > 0 for w in w_0])
    assert (w_1[0] <= 0) and (w_1[2] > 0)
    assert (w_2[0] > 0) and (w_2[2] <= 0)

    # see previous test for explanation
    sum_weights_per_pixel = np.stack(out_weights, axis=1).sum(axis=1)
    np.testing.assert_almost_equal(1, sum_weights_per_pixel)


def test_error_catching_with_s1_grid():
    """Tests proper error handling with improper inputs"""
    N = 10
    M = 11
    P = 12

    # hgt is mesh but lat/lon are 1d
    lon = np.arange(N)
    lat = np.arange(M)
    hgt = np.zeros((P, P))
    dt = datetime.datetime(2023, 1, 1)
    with pytest.raises(ValueError):
        get_s1_azimuth_time_grid(lon, lat, hgt, dt)

    # lon is not 1 or 3d
    lon = np.zeros((N, N, N, N))
    lat = np.arange(M)
    hgt = np.arange(P)
    with pytest.raises(ValueError):
        get_s1_azimuth_time_grid(lon, lat, hgt, dt)


def test_duplicate_orbits(mocker, orbit_paths_for_duplicate_orbit_xml_test):
    hgt = np.linspace(-500, 26158.0385, 20)
    lat = np.linspace(40.647867694896775, 44.445117773316184, 20)
    lon = np.linspace(-74, -79, 20)
    t = datetime.datetime(2023, 3, 23, 23, 0, 28)

    # These outputs are not needed since the orbits are specified above
    mocker.patch('RAiDER.s1_azimuth_timing.get_slc_id_from_point_and_time',
                 side_effect=[['slc_id_0', 'slc_id_1', 'slc_id_2', 'slc_id_3']])

    mocker.patch(
        'RAiDER.s1_azimuth_timing.get_orbits_from_slc_ids_hyp3lib',
        side_effect=[
            [Path(o_path) for o_path in orbit_paths_for_duplicate_orbit_xml_test],
        ]
    )

    time_grid = get_s1_azimuth_time_grid(lon, lat, hgt, t)

    assert time_grid.shape == (len(hgt), len(lat), len(lon))

    assert RAiDER.s1_azimuth_timing.get_slc_id_from_point_and_time.call_count == 1
    assert RAiDER.s1_azimuth_timing.get_orbits_from_slc_ids_hyp3lib.call_count == 1


def test_get_times_for_az():

    # Within 5 minutes of time-step (aka model time) so returns 3 times
    dt = datetime.datetime(2023, 1, 1, 11, 1, 0)
    out = get_times_for_azimuth_interpolation(dt, 1)

    out_expected = [datetime.datetime(2023, 1, 1, 11, 0, 0),
                    datetime.datetime(2023, 1, 1, 12, 0, 0),
                    datetime.datetime(2023, 1, 1, 10, 0, 0)]

    assert out == out_expected

    # Since model time is now 3 hours, we are beyond buffer, so we get 2 times
    out = get_times_for_azimuth_interpolation(dt, 3)
    out_expected = [datetime.datetime(2023, 1, 1, 12, 0, 0),
                    datetime.datetime(2023, 1, 1, 9, 0, 0)]
    assert out == out_expected

    # Similarly return 2 times if we nudge reference time away from buffer
    # When model time is 1 hour
    # Note that if we chose 11:30 we would get 2 dates which would both be admissible
    dt = datetime.datetime(2023, 1, 1, 11, 29, 0)
    out = get_times_for_azimuth_interpolation(dt, 1)

    out_expected = [datetime.datetime(2023, 1, 1, 11, 0, 0),
                    datetime.datetime(2023, 1, 1, 12, 0, 0)]

    assert out == out_expected


def test_error_for_weighting_when_dates_not_unique():
    dates = [datetime.datetime(2023, 1, 1)] * 2
    with pytest.raises(ValueError):
        get_inverse_weights_for_dates(np.zeros((3, 3)),
                                      dates)
