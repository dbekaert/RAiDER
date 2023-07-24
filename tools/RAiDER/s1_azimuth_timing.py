import datetime
import warnings

import asf_search as asf
import hyp3lib
import isce3.ext.isce3 as isce
import numpy as np
import pandas as pd
from RAiDER.losreader import get_orbit as get_isce_orbit
from shapely.geometry import Point


def _asf_query(point: Point,
               start: datetime.datetime,
               end: datetime.datetime) -> list[str]:
    results = asf.geo_search(intersectsWith=point.wkt,
                             processingLevel=asf.PRODUCT_TYPE.SLC,
                             start=start,
                             end=end,
                             maxResults=5
                             )
    slc_ids = [r.properties['sceneName'] for r in results]
    return slc_ids


def get_slc_id_from_point_and_time(lon: float,
                                   lat: float,
                                   dt: datetime.datetime,
                                   buffer_seconds: int = 60) -> str:
    """Obtains a (non-unique) SLC id from the lon/lat and datetime of inputs. The buffere ensures that
    an SLC id is within the queried start/end times. Note an S1 scene takes roughly 30 seconds to acquire.

    Parameters
    ----------
    lon : float
    lat : float
    dt : datetime.datetime
    buffer_seconds : int, optional
        Do not recommend adjusting this, by default 60

    Returns
    -------
    str
        First slc_id returned by asf_search
    """
    point = Point(lon, lat)
    time_delta = datetime.timedelta(seconds=buffer_seconds)
    start = dt - time_delta
    end = dt + time_delta

    slc_ids = _asf_query(point, start, end)
    if not slc_ids:
        raise ValueError('No results found for input lon/lat and datetime')

    return slc_ids[0]


def get_azimuth_time_grid(lon: np.ndarray,
                          lat: np.ndarray,
                          hgt:  np.ndarray,
                          orb: isce.core.Orbit) -> np.ndarray:
    '''
    Source: https://github.com/dbekaert/RAiDER/blob/dev/tools/RAiDER/losreader.py#L601C1-L674C22

    lon, lat, hgt are coordinate arrays (this routine makes a mesh to comute azimuth timing grid)

    Technically, this is "sensor neutral" since it uses an orb object.
    '''

    num_iteration = 30
    residual_threshold = 1.0e-7

    elp = isce.core.Ellipsoid()
    dop = isce.core.LUT2d()
    look = isce.core.LookSide.Right

    hgt_mesh, lat_mesh, lon_mesh = np.meshgrid(hgt, lat, lon,
                                               # indexing keyword argument
                                               # Ensures output dimensions
                                               # align with order the inputs
                                               # height x latitude x longitude
                                               indexing='ij')
    m, n, p = hgt_mesh.shape
    az_arr = np.full((m, n, p),
                     np.datetime64('NaT'),
                     # source: https://stackoverflow.com/a/27469108
                     dtype='datetime64[ms]')

    for ind_0 in range(m):
        for ind_1 in range(n):
            for ind_2 in range(p):

                hgt_pt, lat_pt, lon_pt = (hgt_mesh[ind_0, ind_1, ind_2],
                                          lat_mesh[ind_0, ind_1, ind_2],
                                          lon_mesh[ind_0, ind_1, ind_2])

                input_vec = np.array([np.deg2rad(lon_pt),
                                      np.deg2rad(lat_pt),
                                      hgt_pt])

                aztime, sr = isce.geometry.geo2rdr(
                    input_vec, elp, orb, dop, 0.06, look,
                    threshold=residual_threshold,
                    maxiter=num_iteration,
                    delta_range=10.0)

                rng_seconds = sr / isce.core.speed_of_light
                aztime = aztime + rng_seconds
                aztime_isce = orb.reference_epoch + isce.core.TimeDelta(aztime)
                aztime_np = np.datetime64(aztime_isce.isoformat())
                az_arr[ind_0, ind_1, ind_2] = aztime_np
    return az_arr


def get_s1_azimuth_time_grid(lon: np.ndarray,
                             lat: np.ndarray,
                             hgt:  np.ndarray,
                             dt: datetime.datetime) -> np.ndarray:
    """Based on the lon, lat, hgt (3d cube) - obtains an associated s1 orbit
    file to calculate the azimuth timing across the cube. Requires datetime of acq
    associated to cube.

    Parameters
    ----------
    lon : np.ndarray
        1 dimensional coordinate array
    lat : np.ndarray
        1 dimensional coordinate array
    hgt : np.ndarray
        1 dimensional coordinate array
    dt : datetime.datetime

    Returns
    -------
    np.ndarray
        Cube whose coordinates are hgt x lat x lon with each pixel
    """
    if not all([len(arr.shape) == 1 for arr in [lon, lat, hgt]]):
        raise ValueError('The dimensions of the array must be 1d')

    lon_m = np.mean(lon)
    lat_m = np.mean(lat)

    try:
        slc_id = get_slc_id_from_point_and_time(lon_m, lat_m, dt)
    except ValueError:
        warnings.warn('No slc id found for the given datetime and grid; returning empty grid')
        m, n, p = hgt.shape[0], lat.shape[0], lon.shape[0]
        az_arr = np.full((m, n, p),
                         np.datetime64('NaT'),
                         dtype='datetime64[ms]')
        return az_arr
    orb_file, _ = hyp3lib.get_orb.downloadSentinelOrbitFile(slc_id)
    orb = get_isce_orbit(orb_file, dt, pad=600)

    az_arr = get_azimuth_time_grid(lon, lat, hgt, orb)
    return az_arr


def get_n_closest_datetimes(ref_time: datetime.datetime,
                            n_target_times: int,
                            time_step_hours: int) -> list[datetime.datetime]:
    """Gets n closes times relative to the `round_to_hour_delta` and the
    `ref_time`. Specifically, if one is interetsted in getting 3 closest times
    to say 0, 6, 12, 18 UTC times of a ref time `dt`, then:
    ```
    dt = datetime.datetime(2023, 1, 1, 11, 0, 0)
    get_n_closest_datetimes(dt, 3, 6)
    ```
    gives the desired answer of
    ```
    [datetime.datetime(2023, 1, 1, 12, 0, 0),
     datetime.datetime(2023, 1, 1, 6, 0, 0),
     datetime.datetime(2023, 1, 1, 18, 0, 0)]
    ```

    Parameters
    ----------
    ref_time : datetime.datetime
        Time to round from
    n_times : int
        Number of times to get
    time_step_hours : int
        If 1, then rounds ref_time to nearest hour(s). If 2, then rounds to
        nearest 0, 2, 4, etc. times. Must be divisible by 24 otherwise is
        not consistent across all days.

    Returns
    -------
    list[datetime.datetime]
        List of closest dates ordered by absolute proximity
    """
    iterations = int(np.ceil(n_target_times / 2))
    closest_times = []

    if (24 % time_step_hours) != 0:
        raise ValueError('The time step does not evenly divide 24 hours;'
                         'Time step has period > 1 day and depends when model '
                         'starts')

    ts = pd.Timestamp(ref_time)
    for k in range(iterations):
        ts_0 = pd.Timestamp(ref_time) - pd.Timedelta(hours=(time_step_hours * k))
        ts_1 = pd.Timestamp(ref_time) + pd.Timedelta(hours=(time_step_hours * k))

        t_ceil = ts_0.floor(f'{time_step_hours}H')
        t_floor = ts_1.ceil(f'{time_step_hours}H')

        closest_times.extend([t_ceil, t_floor])
    closest_times = sorted(closest_times, key=lambda ts_rounded: abs(ts - ts_rounded))
    closest_times = [t.to_pydatetime() for t in closest_times]
    closest_times = closest_times[:n_target_times]
    return closest_times


def get_inverse_weights_for_dates(azimuth_time_array: np.ndarray,
                                  dates: list[datetime.datetime],
                                  inverse_regularizer: float = 1e-9,
                                  temporal_window_hours: float = None) -> list[np.ndarray]:
    """Obtains weights according to inverse weighting with respect to the absolute difference between
    azimuth timing array and dates. The output will be a list with length equal to that of dates and
    whose entries are arrays each whose shape matches the azimuth_timing_array.

    Note: we do not do any checking of the dates provided so the inferred `temporal_window_hours` may be incorrect.

    Parameters
    ----------
    azimuth_time_array : np.ndarray
        Array of type `np.datetime64[ms]`
    dates : list[datetime.datetime]
        List of datetimes
    inverse_regularizer : float, optional
        If a `time` in the azimuth time arr equals one of the given dates, then the regularlizer ensures that the value
        `1 / (|date - time| + inverse_regularizer) = weight` is not infinity, by default 1e-9
    temporal_window_hours : float, optional
        Values outside of this are masked from inverse weighted.
        If None, then window is minimum abs difference of dates (inferring the temporal resolution), by default None
        No check of equi-spaced dates are done so not specifying temporal window hours requires dates to be derived
        from valid model time steps

    Returns
    -------
    list[np.ndarray]
        Weighting per pixel with respect to each date
    """
    if not all([isinstance(date, datetime.datetime) for date in dates]):
        raise TypeError('dates must be all datetimes')
    if temporal_window_hours is None:
        temporal_window_seconds = min([abs((date - dates[0]).total_seconds()) for date in dates[1:]])
    else:
        temporal_window_seconds = temporal_window_hours * 60 * 60

    # Get absolute differences
    dates_np = list(map(np.datetime64, dates))
    abs_diff = [np.abs(azimuth_time_array - date) / np.timedelta64(1, 's') for date in dates_np]

    # Get inverse weighting with mask determined by window
    wgts = [1. / (diff + inverse_regularizer) for diff in abs_diff]
    masks = [(diff <= temporal_window_seconds).astype(int) for diff in abs_diff]

    if all([mask.sum() == 0 for mask in masks]):
        raise ValueError('No dates provided are within temporal window')

    # Normalize so that sum of weights is 1
    wgts_masked = [wgt * mask for wgt, mask in zip(wgts, masks)]
    wgts_sum = np.sum(np.stack(wgts_masked, axis=-1), axis=-1)
    wgts_norm = [wgt / wgts_sum for wgt in wgts_masked]
    return wgts_norm
