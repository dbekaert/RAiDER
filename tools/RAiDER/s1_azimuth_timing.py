import datetime
import warnings

import asf_search as asf
import numpy as np
import pandas as pd
from shapely.geometry import Point


try:
    import isce3.ext.isce3 as isce
except ImportError:
    isce = None

from RAiDER.losreader import get_orbit as get_isce_orbit
from RAiDER.s1_orbits import get_orbits_from_slc_ids_hyp3lib


def _asf_query(
    point: Point,
    start: datetime.datetime,
    end: datetime.datetime,
    buffer_degrees: float = 2
) -> list[str]:
    """
    Using a buffer to get as many SLCs covering a given request as.

    Parameters
    ----------
    point : Point
    start : datetime.datetime
    end : datetime.datetime
    buffer_degrees : float, optional

    Returns:
    -------
    list[str]
    """
    results = asf.geo_search(
        intersectsWith=point.buffer(buffer_degrees).wkt,
        processingLevel=asf.PRODUCT_TYPE.SLC,
        start=start,
        end=end,
        maxResults=5,
    )
    slc_ids = [r.properties['sceneName'] for r in results]
    return slc_ids


def get_slc_id_from_point_and_time(
    lon: float,
    lat: float,
    dt: datetime.datetime,
    buffer_seconds: int = 600,
    buffer_deg: float = 2
) -> list:
    """
    Obtains a (non-unique) SLC id from the lon/lat and datetime of inputs. The buffere ensures that
    an SLC id is within the queried start/end times. Note an S1 scene takes roughly 30 seconds to acquire.

    Parameters
    ----------
    lon : float
    lat : float
    dt : datetime.datetime
    buffer_seconds : int, optional
        Do not recommend adjusting this, by default 600, to ensure enough padding for multiple orbit files

    Returns:
    -------
    list
        All slc_ids returned by asf_search
    """
    point = Point(lon, lat)
    time_delta = datetime.timedelta(seconds=buffer_seconds)
    start = dt - time_delta
    end = dt + time_delta

    # Requires buffer of degrees to get several SLCs and ensure we get correct
    # orbit files
    slc_ids = _asf_query(point, start, end, buffer_degrees=buffer_deg)
    if not slc_ids:
        raise ValueError('No results found for input lon/lat and datetime')

    return slc_ids


def get_azimuth_time_grid(
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray,
    hgt_mesh: np.ndarray,
    orb: 'isce.core.Orbit'
) -> np.ndarray:
    """
    Source: https://github.com/dbekaert/RAiDER/blob/dev/tools/RAiDER/losreader.py#L601C1-L674C22

    lon_mesh, lat_mesh, hgt_mesh are coordinate arrays (this routine makes a mesh to comute azimuth timing grid)

    Technically, this is "sensor neutral" since it uses an orb object.
    """
    if isce is None:
        raise ImportError('isce3 is required for this function. Use conda to install isce3`')

    num_iteration = 100
    residual_threshold = 1.0e-7

    elp = isce.core.Ellipsoid()
    dop = isce.core.LUT2d()
    look = isce.core.LookSide.Right

    m, n, p = hgt_mesh.shape
    az_arr = np.full(
        (m, n, p),
        np.datetime64('NaT'),
        # source: https://stackoverflow.com/a/27469108
        dtype='datetime64[ms]',
    )

    for ind_0 in range(m):
        for ind_1 in range(n):
            for ind_2 in range(p):
                hgt_pt, lat_pt, lon_pt = (
                    hgt_mesh[ind_0, ind_1, ind_2],
                    lat_mesh[ind_0, ind_1, ind_2],
                    lon_mesh[ind_0, ind_1, ind_2],
                )

                input_vec = np.array([np.deg2rad(lon_pt), np.deg2rad(lat_pt), hgt_pt])

                aztime, sr = isce.geometry.geo2rdr(
                    input_vec,
                    elp,
                    orb,
                    dop,
                    0.06,
                    look,
                    threshold=residual_threshold,
                    maxiter=num_iteration,
                    delta_range=10.0,
                )

                rng_seconds = sr / isce.core.speed_of_light
                aztime = aztime + rng_seconds
                aztime_isce = orb.reference_epoch + isce.core.TimeDelta(aztime)
                aztime_np = np.datetime64(aztime_isce.isoformat())
                az_arr[ind_0, ind_1, ind_2] = aztime_np
    return az_arr


def get_s1_azimuth_time_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    hgt: np.ndarray,
    dt: datetime.datetime
) -> np.ndarray:
    """Based on the lon, lat, hgt (3d cube) - obtains an associated s1 orbit
    file to calculate the azimuth timing across the cube. Requires datetime of acq
    associated to cube.

    Parameters
    ----------
    lon : np.ndarray
        1 dimensional coordinate array or 3d mesh of coordinates
    lat : np.ndarray
        1 dimensional coordinate array or 3d mesh of coordinates
    hgt : np.ndarray
        1 dimensional coordinate array or 3d mesh of coordinates
    dt : datetime.datetime

    Returns:
    -------
    np.ndarray
        Cube whose coordinates are hgt x lat x lon with each pixel
    """
    dims = [len(c.shape) for c in [lon, lat, hgt]]
    if not all([dim == dims[0] for dim in dims]):
        raise ValueError('All coordinates have same dimension (either 1 or 3 dimensional)')
    if not all([dim in [1, 3] for dim in dims]):
        raise ValueError('Coordinates must be 1d or 3d coordinate arrays')

    if dims[0] == 1:
        hgt_mesh, lat_mesh, lon_mesh = np.meshgrid(
            hgt,
            lat,
            lon,
            # indexing keyword argument
            # Ensures output dimensions
            # align with order the inputs
            # height x latitude x longitude
            indexing='ij',
        )
    else:
        hgt_mesh = hgt
        lat_mesh = lat
        lon_mesh = lon

    try:
        lon_m = np.mean(lon)
        lat_m = np.mean(lat)
        slc_ids = get_slc_id_from_point_and_time(lon_m, lat_m, dt)
    except ValueError:
        warnings.warn('No slc id found for the given datetime and grid; returning empty grid')
        m, n, p = hgt_mesh.shape
        az_arr = np.full((m, n, p), np.datetime64('NaT'), dtype='datetime64[ms]')
        return az_arr

    orb_files = get_orbits_from_slc_ids_hyp3lib(slc_ids)
    orb_files = [str(of) for of in orb_files]

    orb = get_isce_orbit(orb_files, dt, pad=600)
    az_arr = get_azimuth_time_grid(lon_mesh, lat_mesh, hgt_mesh, orb)

    return az_arr


def get_n_closest_datetimes(
    ref_time: datetime.datetime,
    n_target_times: int,
    time_step_hours: int
) -> list[datetime.datetime]:
    """
    Gets n closest times relative to the `round_to_hour_delta` and the
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

    Returns:
    -------
    list[datetime.datetime]
        List of closest dates ordered by absolute proximity. If two dates have same distance to ref_time,
        choose earlier one (more likely to be available)
    """
    iterations = int(np.ceil(n_target_times / 2))
    closest_times = []

    if (24 % time_step_hours) != 0:
        raise ValueError(
            'The time step does not evenly divide 24 hours;'
            'Time step has period > 1 day and depends when model '
            'starts'
        )

    ts = pd.Timestamp(ref_time)
    for k in range(iterations):
        ts_0 = ts - pd.Timedelta(hours=(time_step_hours * k))
        ts_1 = ts + pd.Timedelta(hours=(time_step_hours * k))

        t_ceil = ts_0.floor(f'{time_step_hours}H')
        t_floor = ts_1.ceil(f'{time_step_hours}H')
        # In the event that t_floor == t_ceil for k = 0
        out_times = list(set([t_ceil, t_floor]))
        closest_times.extend(out_times)
    # if 2 times have same distance to ref_time, order times by occurence (earlier comes first)
    closest_times = sorted(closest_times, key=lambda ts_rounded: (abs(ts - ts_rounded), ts_rounded))
    closest_times = [t.to_pydatetime() for t in closest_times]
    closest_times = closest_times[:n_target_times]
    return closest_times


def get_times_for_azimuth_interpolation(
    ref_time: datetime.datetime,
    time_step_hours: int,
    buffer_in_seconds: int = 300
) -> list[datetime.datetime]:
    """Obtains times needed for azimuth interpolation. Filters 3 closests dates from ref_time
    so that all returned dates are within `time_step_hours` + `buffer_in_seconds`.

    This ensures we request dates that are really needed.
    ```
    dt = datetime.datetime(2023, 1, 1, 11, 1, 0)
    get_times_for_azimuth_interpolation(dt, 1)
    ```
    yields
    ```
    [datetime.datetime(2023, 1, 1, 11, 0, 0),
     datetime.datetime(2023, 1, 1, 12, 0, 0),
     datetime.datetime(2023, 1, 1, 10, 0, 0)]
    ```
    whereas
    ```
    dt = datetime.datetime(2023, 1, 1, 11, 30, 0)
    get_times_for_azimuth_interpolation(dt, 1)
    ```
    yields
    ```
    [datetime.datetime(2023, 1, 1, 11, 0, 0),
     datetime.datetime(2023, 1, 1, 12, 0, 0)]
    ```

    Parameters
    ----------
    ref_time : datetime.datetime
        A time of acquisition
    time_step_hours : int
        Weather model time step, should evenly divide 24 hours
    buffer_in_seconds : int, optional
        Buffer for filtering absolute times, by default 300 (or 5 minutes)

    Returns:
    -------
    list[datetime.datetime]
        2 or 3 closest times within 1 time step (plust the buffer) and the reference time
    """
    # Get 3 closest times
    closest_times = get_n_closest_datetimes(ref_time, 3, time_step_hours)

    def filter_time(time: datetime.datetime):
        absolute_time_difference_sec = abs((ref_time - time).total_seconds())
        upper_bound_seconds = time_step_hours * 60 * 60 + buffer_in_seconds
        return absolute_time_difference_sec < upper_bound_seconds

    out_times = list(filter(filter_time, closest_times))
    return out_times


def get_inverse_weights_for_dates(
    azimuth_time_array: np.ndarray,
    dates: list[datetime.datetime],
    inverse_regularizer: float = 1e-9,
    temporal_window_hours: float = None,
) -> list[np.ndarray]:
    """Obtains weights according to inverse weighting with respect to the absolute difference between
    azimuth timing array and dates. The output will be a list with length equal to that of dates and
    whose entries are arrays each whose shape matches the azimuth_timing_array.

    Note: we do not do any checking of the provided dates outside that they are unique so the inferred
    `temporal_window_hours` may be incorrect.

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

    Returns:
    -------
    list[np.ndarray]
        Weighting per pixel with respect to each date
    """
    n_unique_dates = len(set(dates))
    n_dates = len(dates)
    if n_unique_dates != n_dates:
        raise ValueError('Dates provided must be unique')
    if n_dates == 0:
        raise ValueError('No dates provided')

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
    wgts = [1.0 / (diff + inverse_regularizer) for diff in abs_diff]
    masks = [(diff <= temporal_window_seconds).astype(int) for diff in abs_diff]

    if all([mask.sum() == 0 for mask in masks]):
        raise ValueError('No dates provided are within temporal window')

    # Normalize so that sum of weights is 1
    wgts_masked = [wgt * mask for wgt, mask in zip(wgts, masks)]
    wgts_sum = np.sum(np.stack(wgts_masked, axis=-1), axis=-1)
    wgts_norm = [wgt / wgts_sum for wgt in wgts_masked]
    return wgts_norm
