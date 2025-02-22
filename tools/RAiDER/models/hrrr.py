import datetime as dt
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from herbie import Herbie
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box

from RAiDER.logger import logger
from RAiDER.models.model_levels import LEVELS_50_HEIGHTS
from RAiDER.models.weatherModel import TIME_RES, WeatherModel
from RAiDER.utilFcns import round_date


HRRR_CONUS_COVERAGE_POLYGON = Polygon(((-125, 21), (-133, 49), (-60, 49), (-72, 21)))
HRRR_AK_COVERAGE_POLYGON = Polygon(((195, 40), (157, 55), (175, 70), (260, 77), (232, 52)))
HRRR_AK_PROJ = CRS.from_string(
    '+proj=stere +ellps=sphere +a=6371229.0 +b=6371229.0 +lat_0=90 +lon_0=225.0 '
    '+x_0=0.0 +y_0=0.0 +lat_ts=60.0 +no_defs +type=crs'
)
# Source: https://eric.clst.org/tech/usgeojson/
AK_GEO = gpd.read_file(Path(__file__).parent / 'data' / 'alaska.geojson.zip').geometry.unary_union


def check_hrrr_dataset_availability(datetime: dt.datetime) -> bool:
    """Note a file could still be missing within the models valid range."""
    herbie = Herbie(
        datetime,
        model='hrrr',
        product='nat',
        fxx=0,
    )
    return herbie.grib_source is not None


def download_hrrr_file(ll_bounds, DATE, out, model='hrrr', product='nat', fxx=0, verbose=False) -> None:
    """
    Download a HRRR weather model using Herbie.

    Args:
        DATE (Python datetime)  - Datetime as a Python datetime. Herbie will automatically return the closest valid time,
                                    which is currently hourly.
        out (string)            - output location as a string
        model (string)          - model can be "hrrr" or "hrrrak"
        product (string)        - 'prs' for pressure levels, 'nat' for native levels
        fxx (int)               - forecast time in hours. Can be up to 48 for 00/06/12/18
        verbose (bool)          - True for extra printout of information

    Returns:
        None, writes data to a netcdf file
    """
    herbie = Herbie(
        DATE.strftime('%Y-%m-%d %H:%M'),
        model=model,
        product=product,
        fxx=fxx,
        overwrite=False,
        verbose=True,
        save_dir=Path(os.path.dirname(out)),
    )

    # Iterate through the list of datasets
    try:
        ds_list = herbie.xarray(':(SPFH|PRES|TMP|HGT):', verbose=verbose)
    except ValueError as e:
        logger.error(e)
        raise

    ds_out = None
    # Note order coord names are request for `test_HRRR_ztd` matters
    # when both coord names are retreived by Herbie is ds_list possibly in
    # Different orders on different machines; `hybrid` is what is expected for the test.
    ds_list_filt_0 = [ds for ds in ds_list if 'hybrid' in ds._coord_names]
    ds_list_filt_1 = [ds for ds in ds_list if 'isobaricInhPa' in ds._coord_names]
    if ds_list_filt_0:
        ds_out = ds_list_filt_0[0]
        coord = 'hybrid'
    #  I do not think that this coord name will result in successful processing nominally as variables are
    #  gh,gribfile_projection for test_HRRR_ztd
    elif ds_list_filt_1:
        ds_out = ds_list_filt_1[0]
        coord = 'isobaricInhPa'
    else:
        raise RuntimeError('Herbie did not obtain an HRRR dataset with the expected layers and coordinates')

    # subset the full file by AOI
    x_min, x_max, y_min, y_max = get_bounds_indices(
        ll_bounds,
        ds_out.latitude.to_numpy(),
        ds_out.longitude.to_numpy(),
    )

    # bookkeepping
    ds_out = ds_out.rename({'gh': 'z', coord: 'levels'})

    # projection information
    ds_out['proj'] = 0
    for k, v in CRS.from_user_input(ds_out.herbie.crs).to_cf().items():
        ds_out.proj.attrs[k] = v
    for var in ds_out.data_vars:
        ds_out[var].attrs['grid_mapping'] = 'proj'

    # pull the grid information
    proj = CRS.from_cf(ds_out['proj'].attrs)
    t = Transformer.from_crs(4326, proj, always_xy=True)
    xl, yl = t.transform(ds_out['longitude'].values, ds_out['latitude'].values)
    WW, EE = ds_out['longitude'].values.min(), ds_out['longitude'].values.max()
    SS, NN = ds_out['latitude'].values.min(), ds_out['latitude'].values.max()
    W, E, S, N = np.nanmin(xl), np.nanmax(xl), np.nanmin(yl), np.nanmax(yl)

    grid_x = 3000  # meters
    grid_y = 3000  # meters
    xs = np.arange(W, E + grid_x / 2, grid_x)
    ys = np.arange(S, N + grid_y / 2, grid_y)

    try:
        ds_out['x'] = xs
    except Exception as e:
        print(ds_out)
        print(xs.shape)
        print(e)
        raise Exception(f'\n\nError {proj}, {[WW, SS, EE, NN]}\n----------------------------------\n\n')
    ds_out['y'] = ys
    ds_sub = ds_out.isel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    ds_sub.to_netcdf(out, engine='netcdf4')


def get_bounds_indices(SNWE, lats, lons):
    """Convert SNWE lat/lon bounds to index bounds."""
    # Unpack the bounds and find the relevent indices
    S, N, W, E = SNWE

    # Need to account for crossing the international date line
    if W < E:
        m1 = (S <= lats) & (N >= lats) & (W <= lons) & (E >= lons)
    else:
        raise ValueError(
            'Longitude is either flipped or you are crossing the international date line;'
            + 'if the latter please give me longitudes from 0-360'
        )

    if np.sum(m1) == 0:
        lons = np.mod(lons, 360)
        W, E = np.mod([W, E], 360)
        m1 = (S <= lats) & (N >= lats) & (W <= lons) & (E >= lons)
        if np.sum(m1) == 0:
            raise RuntimeError('Area of Interest has no overlap with the HRRR model available extent')

    # Y extent
    shp = lats.shape
    m1_y = np.argwhere(np.sum(m1, axis=1) != 0)
    y_min = max(m1_y[0][0], 0)
    y_max = min(m1_y[-1][0], shp[0])
    m1_y = None

    # X extent
    m1_x = np.argwhere(np.sum(m1, axis=0) != 0)
    x_min = max(m1_x[0][0], 0)
    x_max = min(m1_x[-1][0], shp[1])
    m1_x = None
    m1 = None

    return x_min, x_max, y_min, y_max


def load_weather_hrrr(filename):
    """Loads a weather model from a HRRR file."""
    # read data from the netcdf file
    ds = xr.open_dataset(filename, engine='netcdf4')
    # Pull the relevant data from the file
    pres = ds['pres'].values.transpose(1, 2, 0)
    xArr = ds['x'].values
    yArr = ds['y'].values
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    temps = ds['t'].values.transpose(1, 2, 0)
    qs = ds['q'].values.transpose(1, 2, 0)
    geo_hgt = ds['z'].values.transpose(1, 2, 0)

    proj = CRS.from_cf(ds['proj'].attrs)

    lons[lons > 180] -= 360

    # data cube format should be lats,lons,heights
    _xs = np.broadcast_to(xArr[np.newaxis, :, np.newaxis], geo_hgt.shape)
    _ys = np.broadcast_to(yArr[:, np.newaxis, np.newaxis], geo_hgt.shape)

    return _xs, _ys, lons, lats, qs, temps, pres, geo_hgt, proj


class HRRR(WeatherModel):
    def __init__(self) -> None:
        # initialize a weather model
        super().__init__()

        self._humidityType = 'q'
        self._model_level_type = 'pl'  # Default, pressure levels are 'pl'
        self._expver = '0001'
        self._classname = 'hrrr'
        self._dataset = 'hrrr'

        self._time_res = TIME_RES[self._dataset.upper()]

        # Tuple of min/max years where data is available.
        self._valid_range = (
            dt.datetime(2016, 7, 15).replace(tzinfo=dt.timezone(offset=dt.timedelta())),
            dt.datetime.now(dt.timezone.utc),
        )
        self._lag_time = dt.timedelta(hours=3)  # Availability lag time in days

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # 3 km horizontal grid spacing
        self._lat_res = 3.0 / 111
        self._lon_res = 3.0 / 111
        self._x_res = 3.0
        self._y_res = 3.0

        self._Nproc = 1
        self._Name = 'HRRR'
        self._Npl = 0
        self.files = None
        self._bounds = None

        # Projection
        # NOTE: The HRRR projection will get read directly from the downloaded weather model file; however,
        # we also define it here so that the projection can be used without downloading any data. This is
        # used for consistency with the other weather models and allows for some nice features, such as
        # buffering.

        # See https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb and code lower down
        # '262.5:38.5:38.5:38.5 237.280472:1799:3000.00 21.138123:1059:3000.00'
        # 'lov:latin1:latin2:latd lon1:nx:dx lat1:ny:dy'
        # LCC parameters
        lon0 = 262.5
        lat0 = 38.5
        lat1 = 38.5
        lat2 = 38.5
        x0 = 0
        y0 = 0
        earth_radius = 6371229
        self._proj = CRS(
            f'+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} '
            f'+lon_0={lon0} +x_0={x0} +y_0={y0} +a={earth_radius} '
            f'+b={earth_radius} +units=m +no_defs'
        )
        self._valid_bounds = HRRR_CONUS_COVERAGE_POLYGON
        self.setLevelType('nat')

    def __model_levels__(self):
        self._levels = 50
        self._zlevels = np.flipud(LEVELS_50_HEIGHTS)

    def __pressure_levels__(self):
        raise NotImplementedError('Pressure levels do not go high enough for HRRR.')

    def _fetch(self, out) -> None:
        """Fetch weather model data from HRRR."""
        self._files = out
        corrected_DT = round_date(self._time, dt.timedelta(hours=self._time_res))
        self.checkTime(corrected_DT)
        if not corrected_DT == self._time:
            logger.info('Rounded given datetime from  %s to %s', self._time, corrected_DT)

        # HRRR uses 0-360 longitude, so we need to convert the bounds to that
        bounds = self._ll_bounds.copy()
        bounds[2:] = np.mod(bounds[2:], 360)

        download_hrrr_file(bounds, corrected_DT, out, 'hrrr', self._model_level_type)

    def load_weather(self, f=None, *args, **kwargs) -> None:
        """
        Load a weather model into a python weatherModel object, from self.files if no
        filename is passed.
        """
        if f is None:
            f = self.files[0] if isinstance(self.files, list) else self.files

        _xs, _ys, _lons, _lats, qs, temps, pres, geo_hgt, proj = load_weather_hrrr(f)

        # convert geopotential height to geometric height
        self._get_heights(_lats, geo_hgt)

        self._t = temps
        self._q = qs
        self._p = pres
        self._xs = _xs
        self._ys = _ys
        self._lats = _lats
        self._lons = _lons
        self._proj = proj

    def checkValidBounds(self, ll_bounds: np.ndarray) -> None:
        """
        Checks whether the given bounding box is valid for the HRRR or HRRRAK
        (i.e., intersects with the model domain at all).

        Args:
        ll_bounds : np.ndarray
        """
        S, N, W, E = ll_bounds
        aoi = box(W, S, E, N)
        if self._valid_bounds.contains(aoi):
            Mod = self

        elif aoi.intersects(self._valid_bounds):
            Mod = self
            logger.critical('The HRRR weather model extent does not completely cover your AOI!')

        else:
            Mod = HRRRAK()
            # valid bounds are in 0->360 to account for dateline crossing
            W, E = np.mod([W, E], 360)
            aoi = box(W, S, E, N)
            if Mod._valid_bounds.contains(aoi):
                pass
            elif aoi.intersects(Mod._valid_bounds):
                logger.critical('The HRRR-AK weather model extent does not completely cover your AOI!')

            else:
                raise ValueError('The requested location is unavailable for HRRR')


class HRRRAK(WeatherModel):
    def __init__(self) -> None:
        # The HRRR-AK model has a few different parameters than HRRR-CONUS.
        # These will get used if a user requests a bounding box in Alaska
        super().__init__()

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # 3 km horizontal grid spacing
        self._lat_res = 3.0 / 111
        self._lon_res = 3.0 / 111
        self._x_res = 3.0
        self._y_res = 3.0

        self._Nproc = 1
        self._Npl = 0
        self.files = None
        self._bounds = None

        self._classname = 'hrrrak'
        self._dataset = 'hrrrak'
        self._Name = 'HRRR-AK'
        self._time_res = TIME_RES['HRRR-AK']
        self._valid_range = (
            dt.datetime(2018, 7, 13).replace(tzinfo=dt.timezone(offset=dt.timedelta())),
            dt.datetime.now(dt.timezone.utc),
        )
        self._lag_time = dt.timedelta(hours=3)
        self._valid_bounds = HRRR_AK_COVERAGE_POLYGON
        # The projection information gets read directly from the  weather model file but we
        # keep this here for object instantiation.
        self._proj = HRRR_AK_PROJ
        self.setLevelType('nat')

    def __model_levels__(self):
        self._levels = 50
        self._zlevels = np.flipud(LEVELS_50_HEIGHTS)

    def __pressure_levels__(self):
        raise NotImplementedError(
            'hrrr.py: Revisit whether or not pressure levels from HRRR can be used for delay calculations; they do not go high enough compared to native model levels.'
        )

    def _fetch(self, out) -> None:
        bounds = self._ll_bounds.copy()
        bounds[2:] = np.mod(bounds[2:], 360)
        corrected_DT = round_date(self._time, dt.timedelta(hours=self._time_res))
        self.checkTime(corrected_DT)
        if not corrected_DT == self._time:
            logger.info(f'Rounded given datetime from {self._time} to {corrected_DT}')

        download_hrrr_file(bounds, corrected_DT, out, 'hrrrak', self._model_level_type)

    def load_weather(self, f=None, *args, **kwargs) -> None:
        if f is None:
            f = self.files[0] if isinstance(self.files, list) else self.files
        _xs, _ys, _lons, _lats, qs, temps, pres, geo_hgt, proj = load_weather_hrrr(f)

        # correct for latitude
        self._get_heights(_lats, geo_hgt)

        self._t = temps
        self._q = qs
        self._p = pres
        self._xs = _xs
        self._ys = _ys
        self._lats = _lats
        self._lons = _lons
        self._proj = proj
