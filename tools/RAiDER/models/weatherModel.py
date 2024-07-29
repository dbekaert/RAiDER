import datetime
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import xarray
from pyproj import CRS
from shapely.affinity import translate
from shapely.geometry import box
from shapely.ops import unary_union

from RAiDER import utilFcns as util
from RAiDER.constants import _ZMIN, _ZREF, _g0
from RAiDER.interpolate import interpolate_along_axis
from RAiDER.interpolator import fillna3D
from RAiDER.logger import logger
from RAiDER.models import plotWeather as plots
from RAiDER.models.customExceptions import DatetimeOutsideRange
from RAiDER.utilFcns import calcgeoh, clip_bbox, robmax, robmin, transform_coords


TIME_RES = {
    'GMAO': 3,
    'ECMWF': 1,
    'HRES': 6,
    'HRRR': 1,
    'WRF': 1,
    'NCMR': 1,
    'HRRR-AK': 3,
}


class WeatherModel(ABC):
    """Implement a generic weather model for getting estimated SAR delays."""

    _dataset: Optional[str]

    def __init__(self) -> None:
        # Initialize model-specific constants/parameters
        self._k1 = None
        self._k2 = None
        self._k3 = None
        self._humidityType = 'q'
        self._a = []
        self._b = []

        self.files = None

        self._time_res = None  # time resolution of the weather model in hours

        self._lon_res = None
        self._lat_res = None
        self._x_res = None
        self._y_res = None

        self._classname = None
        self._dataset = None
        self._Name = None
        self._wmLoc = None

        self._model_level_type = 'ml'

        self._valid_range = (
            datetime.datetime(1900, 1, 1).replace(tzinfo=datetime.timezone(offset=datetime.timedelta())),
            datetime.datetime.now(datetime.timezone.utc).date(),
        )  # Tuple of min/max years where data is available.
        self._lag_time = datetime.timedelta(days=30)  # Availability lag time in days
        self._time = None
        self._bbox = None

        # Define fixed constants
        self._R_v = 461.524
        self._R_d = 287.06  # in our original code this was 287.053
        self._g0 = _g0  # gravity constant
        self._zmin = _ZMIN  # minimum integration height
        self._zmax = _ZREF  # max integration height
        self._proj = None

        # setup data structures
        self._levels = []
        self._xs = np.empty((1, 1, 1))  # Use generic x/y/z instead of lon/lat/height
        self._ys = np.empty((1, 1, 1))
        self._zs = np.empty((1, 1, 1))

        self._lats = None
        self._lons = None
        self._ll_bounds = None
        self._valid_bounds = box(-180, -90, 180, 90)  # Shapely box with WSEN bounds

        self._p = None
        self._q = None
        self._rh = None
        self._t = None
        self._e = None
        self._wet_refractivity = None
        self._hydrostatic_refractivity = None
        self._wet_ztd = None
        self._hydrostatic_ztd = None

    def __str__(self) -> str:
        string = '\n'
        string += '======Weather Model class object=====\n'
        string += f'Weather model time: {self._time}\n'
        string += f'Latitude resolution: {self._lat_res}\n'
        string += f'Longitude resolution: {self._lon_res}\n'
        string += f'Native projection: {self._proj}\n'
        string += f'ZMIN: {self._zmin}\n'
        string += f'ZMAX: {self._zmax}\n'
        string += f'k1 = {self._k1}\n'
        string += f'k2 = {self._k2}\n'
        string += f'k3 = {self._k3}\n'
        string += f'Humidity type = {self._humidityType}\n'
        string += '=====================================\n'
        string += f'Class name: {self._classname}\n'
        string += f'Dataset: {self._dataset}\n'
        string += '=====================================\n'
        string += f'A: {self._a}\n'
        string += f'B: {self._b}\n'
        if self._p is not None:
            string += 'Number of points in Lon/Lat = {}/{}\n'.format(*self._p.shape[:2])
            string += f'Total number of grid points (3D): {np.prod(self._p.shape)}\n'
        if self._xs.size == 0:
            string += f'Minimum/Maximum y: {robmin(self._ys): 4.2f}/{robmax(self._ys): 4.2f}\n'
            string += f'Minimum/Maximum x: {robmin(self._xs): 4.2f}/{robmax(self._xs): 4.2f}\n'
            string += f'Minimum/Maximum zs/heights: {robmin(self._zs): 10.2f}/{robmax(self._zs): 10.2f}\n'

        string += '=====================================\n'
        return string

    def Model(self):
        return self._Name

    def dtime(self):
        return self._time_res

    def getLLRes(self):
        return np.max([self._lat_res, self._lon_res])

    def fetch(self, out, time) -> None:
        """
        Checks the input datetime against the valid date range for the model and then
        calls the model _fetch routine.

        Args:
        ----------
        out -
        ll_bounds - 4 x 1 array, SNWE
        time = UTC datetime
        """
        self.checkTime(time)
        self.setTime(time)

        # write the error raised by the weather model API to the log
        try:
            self._fetch(out)
        except Exception as E:
            logger.exception(E)
            raise

    @abstractmethod
    def _fetch(self, out):
        """Placeholder method. Should be implemented in each weather model type class."""
        pass

    def getTime(self):
        return self._time

    def setTime(self, time, fmt='%Y-%m-%dT%H:%M:%S') -> None:
        """Set the time for a weather model."""
        if isinstance(time, str):
            self._time = datetime.datetime.strptime(time, fmt)
        elif isinstance(time, datetime.datetime):
            self._time = time
        else:
            raise ValueError('"time" must be a string or a datetime object')
        if self._time.tzinfo is None:
            self._time = self._time.replace(tzinfo=datetime.timezone(offset=datetime.timedelta()))

    def get_latlon_bounds(self):
        return self._ll_bounds

    def set_latlon_bounds(self, ll_bounds, Nextra=2, output_spacing=None) -> None:
        """
        Need to correct lat/lon bounds because not all of the weather models have valid
        data exactly bounded by -90/90 (lats) and -180/180 (lons); for GMAO and MERRA2,
        need to adjust the longitude higher end with an extra buffer; for other models,
        the exact bounds are close to -90/90 (lats) and -180/180 (lons) and thus can be
        rounded to the above regions (either in the downloading-file API or subsetting-
        data API) without problems.
        """
        ex_buffer_lon_max = 0.0

        if self._Name in 'HRRR HRRR-AK HRES'.split():
            Nextra = 6  # have a bigger buffer

        else:
            ex_buffer_lon_max = self._lon_res

        # At boundary lats and lons, need to modify Nextra buffer so that the lats and lons do not exceed the boundary
        S, N, W, E = ll_bounds

        # Adjust bounds if they get near the poles or IDL
        pixlat, pixlon = Nextra * self._lat_res, Nextra * self._lon_res

        S = np.max([S - pixlat, -90.0 + pixlat])
        N = np.min([N + pixlat, 90.0 - pixlat])
        W = np.max([W - (pixlon + ex_buffer_lon_max), -180.0 + (pixlon + ex_buffer_lon_max)])
        E = np.min([E + (pixlon + ex_buffer_lon_max), 180.0 - pixlon - ex_buffer_lon_max])
        if output_spacing is not None:
            S, N, W, E = clip_bbox([S, N, W, E], output_spacing)

        self._ll_bounds = np.array([S, N, W, E])

    def get_wmLoc(self):
        """Get the path to the direct with the weather model files."""
        if self._wmLoc is None:
            wmLoc = os.path.join(os.getcwd(), 'weather_files')
        else:
            wmLoc = self._wmLoc
        return wmLoc

    def set_wmLoc(self, weather_model_directory: str) -> None:
        """Set the path to the directory with the weather model files."""
        self._wmLoc = weather_model_directory

    def load(self, *args, _zlevels=None, **kwargs):
        """
        Calls the load_weather method. Each model class should define a load_weather
        method appropriate for that class. 'args' should be one or more filenames.
        """
        # If the weather file has already been processed, do nothing
        outLoc = self.get_wmLoc()
        path_wm_raw = make_raw_weather_data_filename(outLoc, self.Model(), self.getTime())
        self._out_name = self.out_file(outLoc)

        if os.path.exists(self._out_name):
            return self._out_name
        else:
            # Load the weather just for the query points
            self.load_weather(f=path_wm_raw, *args, **kwargs)

            # Process the weather model data
            self._find_e()
            self._uniform_in_z(_zlevels=_zlevels)
            self._checkForNans()
            self._get_wet_refractivity()
            self._get_hydro_refractivity()
            self._adjust_grid(self.get_latlon_bounds())

            # Compute Zenith delays at the weather model grid nodes
            self._getZTD()
            return None

    @abstractmethod
    def load_weather(self, *args, **kwargs):
        """Placeholder method. Should be implemented in each weather model type class."""
        pass

    def plot(self, plotType='pqt', savefig=True):
        """Plotting method. Valid plot types are 'pqt'."""
        if plotType == 'pqt':
            plot = plots.plot_pqt(self, savefig)
        elif plotType == 'wh':
            plot = plots.plot_wh(self, savefig)
        else:
            raise RuntimeError(f'WeatherModel.plot: No plotType named {plotType}')
        return plot

    def checkTime(self, time) -> None:
        """
        Checks the time against the lag time and valid date range for the given model type.

        Parameters:
            time    - Python datetime object

        Raises:
            Different errors depending on the issue
        """
        start_time = self._valid_range[0]
        end_time = self._valid_range[1]

        if not isinstance(time, datetime.datetime):
            raise ValueError(f'"time" should be a Python datetime object, instead it is {time}')

        # This is needed because Python now gets angry if you try to compare non-timezone-aware
        # objects with time-zone aware objects.
        time = time.replace(tzinfo=datetime.timezone(offset=datetime.timedelta()))

        logger.info('Weather model %s is available from %s to %s', self.Model(), start_time, end_time)
        if time < start_time:
            raise DatetimeOutsideRange(self.Model(), time)

        if end_time < time:
            raise DatetimeOutsideRange(self.Model(), time)

        # datetime.datetime.utcnow() is deprecated because Python developers
        # want everyone to use timezone-aware datetimes.
        if time > datetime.datetime.now(datetime.timezone.utc) - self._lag_time:
            raise DatetimeOutsideRange(self.Model(), time)

    def setLevelType(self, levelType) -> None:
        """Set the level type to model levels or pressure levels."""
        if levelType in 'ml pl nat prs'.split():
            self._model_level_type = levelType
        else:
            raise RuntimeError(f'Level type {levelType} is not recognized')

        if levelType in 'ml nat'.split():
            self.__model_levels__()
        else:
            self.__pressure_levels__()

    def _convertmb2Pa(self, pres):
        """Convert pressure in millibars to Pascals."""
        return 100 * pres

    def _get_heights(self, lats, geo_hgt, geo_ht_fill=np.nan) -> None:
        """Transform geo heights to WGS84 ellipsoidal heights."""
        geo_ht_fix = np.where(geo_hgt != geo_ht_fill, geo_hgt, np.nan)
        lats_full = np.broadcast_to(lats[..., np.newaxis], geo_ht_fix.shape)
        self._zs = util.geo_to_ht(lats_full, geo_ht_fix)

    def _find_e(self) -> None:
        """Check the type of e-calculation needed."""
        if self._humidityType == 'rh':
            self._find_e_from_rh()
        elif self._humidityType == 'q':
            self._find_e_from_q()
        else:
            raise RuntimeError('Not a valid humidity type')
        self._rh = None
        self._q = None

    def _find_e_from_q(self) -> None:
        """Calculate e, partial pressure of water vapor."""
        svp = find_svp(self._t)
        # We have q = w/(w + 1), so w = q/(1 - q)
        w = self._q / (1 - self._q)
        self._e = w * self._R_v * (self._p - svp) / self._R_d

    def _find_e_from_rh(self) -> None:
        """Calculate partial pressure of water vapor."""
        svp = find_svp(self._t)
        self._e = self._rh / 100 * svp

    def _get_wet_refractivity(self) -> None:
        """Calculate the wet delay from pressure, temperature, and e."""
        self._wet_refractivity = self._k2 * self._e / self._t + self._k3 * self._e / self._t**2

    def _get_hydro_refractivity(self) -> None:
        """Calculate the hydrostatic delay from pressure and temperature."""
        self._hydrostatic_refractivity = self._k1 * self._p / self._t

    def getWetRefractivity(self):
        return self._wet_refractivity

    def getHydroRefractivity(self):
        return self._hydrostatic_refractivity

    def _adjust_grid(self, ll_bounds=None) -> None:
        """
        This function pads the weather grid with a level at self._zmin, if
        it does not already go that low.
        <<The functionality below has been removed.>>
        <<It also removes levels that are above self._zmax, since they are not needed.>>
        """
        if self._zmin < np.nanmin(self._zs):
            # first add in a new layer at zmin
            self._zs = np.insert(self._zs, 0, self._zmin)

            self._p = util.padLower(self._p)
            self._t = util.padLower(self._t)
            self._e = util.padLower(self._e)
            self._wet_refractivity = util.padLower(self._wet_refractivity)
            self._hydrostatic_refractivity = util.padLower(self._hydrostatic_refractivity)
            if ll_bounds is not None:
                self._trimExtent(ll_bounds)

    def _getZTD(self) -> None:
        """
        Compute the full slant tropospheric delay for each weather model grid node, using the reference
        height zref.
        """
        wet = self.getWetRefractivity()
        hydro = self.getHydroRefractivity()

        # Get the integrated ZTD
        wet_total, hydro_total = np.zeros(wet.shape), np.zeros(hydro.shape)
        for level in range(wet.shape[2]):
            wet_total[..., level] = 1e-6 * np.trapz(wet[..., level:], x=self._zs[level:], axis=2)
            hydro_total[..., level] = 1e-6 * np.trapz(hydro[..., level:], x=self._zs[level:], axis=2)
        self._hydrostatic_ztd = hydro_total
        self._wet_ztd = wet_total

    def _getExtent(self, lats, lons):
        """Get the bounding box around a set of lats/lons."""
        if (lats.size == 1) & (lons.size == 1):
            return [lats - self._lat_res, lats + self._lat_res, lons - self._lon_res, lons + self._lon_res]
        elif (lats.size > 1) & (lons.size > 1):
            return [np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)]
        elif lats.size == 1:
            return [lats - self._lat_res, lats + self._lat_res, np.nanmin(lons), np.nanmax(lons)]
        elif lons.size == 1:
            return [np.nanmin(lats), np.nanmax(lats), lons - self._lon_res, lons + self._lon_res]
        else:
            raise RuntimeError('Not a valid lat/lon shape')

    @property
    def bbox(self) -> list:
        """
        Obtains the bounding box of the weather model in lat/lon CRS.

        Returns:
        -------
        list
            xmin, ymin, xmax, ymax

        Raises:
        ------
        ValueError
           When `self.files` is None.
        """
        if self._bbox is None:
            path_weather_model = self.out_file(self.get_wmLoc())
            if not os.path.exists(path_weather_model):
                raise ValueError('Need to save cropped weather model as netcdf')

            with xarray.load_dataset(path_weather_model) as ds:
                try:
                    xmin, xmax = ds.x.min(), ds.x.max()
                    ymin, ymax = ds.y.min(), ds.y.max()
                except:
                    xmin, xmax = ds.longitude.min(), ds.longitude.max()
                    ymin, ymax = ds.latitude.min(), ds.latitude.max()

            wm_proj = self._proj
            xs, ys = [xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax]
            lons, lats = transform_coords(wm_proj, CRS(4326), xs, ys)
            # projected weather models may not be aligned N/S
            # should only matter for warning messages
            W, E = np.min(lons), np.max(lons)
            # S, N = np.sort([lats[np.argmin(lons)], lats[np.argmax(lons)]])
            S, N = np.min(lats), np.max(lats)
            self._bbox = W, S, E, N

        return self._bbox

    def checkValidBounds(
        self,
        ll_bounds: np.ndarray,
    ) -> None:
        """Check whether the given bounding box is valid for the model (i.e., intersects with the model domain at all).

        Args:
        ll_bounds : np.ndarray
        """
        S, N, W, E = ll_bounds
        if not box(W, S, E, N).intersects(self._valid_bounds):
            raise ValueError(f'The requested location is unavailable for {self._Name}')

    def checkContainment(self, ll_bounds, buffer_deg: float = 1e-5) -> bool:
        """ "
        Checks containment of weather model bbox of outLats and outLons
        provided.

        Args:
        ----------
        weather_model : WeatherModel
        ll_bounds: an array of floats (SNWE) demarcating bbox of targets
        buffer_deg : float
            For x-translates for extents that lie outside of world bounding box,
            this ensures that translates have some overlap. The default is 1e-5
            or ~11.1 meters.

        Returns:
        -------
        bool
           True if weather model contains bounding box of OutLats and outLons
           and False otherwise.
        """
        ymin_input, ymax_input, xmin_input, xmax_input = ll_bounds
        input_box = box(xmin_input, ymin_input, xmax_input, ymax_input)
        xmin, ymin, xmax, ymax = self.bbox
        weather_model_box = box(xmin, ymin, xmax, ymax)
        world_box = box(-180, -90, 180, 90)

        # Logger
        input_box_str = [f'{x:1.2f}' for x in [xmin_input, ymin_input, xmax_input, ymax_input]]
        weath_box_str = [f'{x:1.2f}' for x in [xmin, ymin, xmax, ymax]]

        weath_box_str = ', '.join(weath_box_str)
        input_box_str = ', '.join(input_box_str)

        logger.info(f'Extent of the weather model is (xmin, ymin, xmax, ymax):' f'{weath_box_str}')
        logger.info(f'Extent of the input is (xmin, ymin, xmax, ymax): ' f'{input_box_str}')

        # If the bounding box goes beyond the normal world extents
        # Look at two x-translates, buffer them, and take their union.
        if not world_box.contains(weather_model_box):
            logger.info(
                'Considering x-translates of weather model +/-360 as bounding box outside of -180, -90, 180, 90'
            )
            translates = [
                weather_model_box.buffer(buffer_deg),
                translate(weather_model_box, xoff=360).buffer(buffer_deg),
                translate(weather_model_box, xoff=-360).buffer(buffer_deg),
            ]
            weather_model_box = unary_union(translates)

        return weather_model_box.contains(input_box)

    def _isOutside(self, extent1, extent2) -> bool:
        """
        Determine whether any of extent1 lies outside extent2.
        extent1/2 should be a list containing [lower_lat, upper_lat, left_lon, right_lon].
        """
        t1 = extent1[0] < extent2[0]
        t2 = extent1[1] > extent2[1]
        t3 = extent1[2] < extent2[2]
        t4 = extent1[3] > extent2[3]
        return np.any([t1, t2, t3, t4])

    def _trimExtent(self, extent) -> None:
        """Get the bounding box around a set of lats/lons."""
        lat = self._lats.copy()
        lon = self._lons.copy()
        lat[np.isnan(lat)] = np.nanmean(lat)
        lon[np.isnan(lon)] = np.nanmean(lon)
        mask = (lat >= extent[0]) & (lat <= extent[1]) & (lon >= extent[2]) & (lon <= extent[3])
        ma1 = np.sum(mask, axis=1).astype('bool')
        ma2 = np.sum(mask, axis=0).astype('bool')
        if np.sum(ma1) == 0 and np.sum(ma2) == 0:
            # Don't need to remove any points
            return

        # indices of the part of the grid to keep
        ny, nx, nz = self._p.shape
        index1 = max(np.arange(len(ma1))[ma1][0] - 2, 0)
        index2 = min(np.arange(len(ma1))[ma1][-1] + 2, ny)
        index3 = max(np.arange(len(ma2))[ma2][0] - 2, 0)
        index4 = min(np.arange(len(ma2))[ma2][-1] + 2, nx)

        # subset around points of interest
        self._lons = self._lons[index1:index2, index3:index4]
        self._lats = self._lats[index1:index2, index3:index4]
        self._xs = self._xs[index3:index4]
        self._ys = self._ys[index1:index2]
        self._p = self._p[index1:index2, index3:index4, ...]
        self._t = self._t[index1:index2, index3:index4, ...]
        self._e = self._e[index1:index2, index3:index4, ...]

        self._wet_refractivity = self._wet_refractivity[index1:index2, index3:index4, ...]
        self._hydrostatic_refractivity = self._hydrostatic_refractivity[index1:index2, index3:index4, :]

    def _calculategeoh(self, z, lnsp):
        """
        Function to calculate pressure, geopotential, and geopotential height
        from the surface pressure and model levels provided by a weather model.
        The model levels are numbered from the highest eleveation to the lowest.
        Inputs:
            self - weather model object with parameters a, b defined
            z    - 3-D array of surface heights for the location(s) of interest
            lnsp - log of the surface pressure
        Outputs:
            geopotential - The geopotential in units of height times acceleration
            pressurelvs  - The pressure at each of the model levels for each of
                           the input points
            geoheight    - The geopotential heights
        """
        return calcgeoh(lnsp, self._t, self._q, z, self._a, self._b, self._R_d, self._levels)

    def getProjection(self):
        """Returns: the native weather projection, which should be a pyproj object."""
        return self._proj

    def getPoints(self):
        return self._xs.copy(), self._ys.copy(), self._zs.copy()

    def _uniform_in_z(self, _zlevels=None) -> None:
        """Interpolate all variables to a regular grid in z."""
        nx, ny = self._p.shape[:2]

        # new regular z-spacing
        if _zlevels is None:
            try:
                _zlevels = self._zlevels
            except BaseException:
                _zlevels = np.nanmean(self._zs, axis=(0, 1))

        new_zs = np.tile(_zlevels, (nx, ny, 1))

        # re-assign values to the uniform z
        self._t = interpolate_along_axis(self._zs, self._t, new_zs, axis=2, fill_value=np.nan).astype(np.float32)
        self._p = interpolate_along_axis(self._zs, self._p, new_zs, axis=2, fill_value=np.nan).astype(np.float32)
        self._e = interpolate_along_axis(self._zs, self._e, new_zs, axis=2, fill_value=np.nan).astype(np.float32)

        self._zs = _zlevels
        self._xs = np.unique(self._xs)
        self._ys = np.unique(self._ys)

    def _checkForNans(self) -> None:
        """Fill in NaN-values."""
        self._p = fillna3D(self._p)
        self._t = fillna3D(self._t, fill_value=1e16)  # to avoid division by zero later on
        self._e = fillna3D(self._e)

    def out_file(self, outLoc):
        f = make_weather_model_filename(
            self._Name,
            self._time,
            self._ll_bounds,
        )
        return os.path.join(outLoc, f)

    def filename(self, time=None, outLoc='weather_files'):
        """Create a filename to store the weather model."""
        os.makedirs(outLoc, exist_ok=True)

        if time is None:
            if self._time is None:
                raise ValueError('Time must be specified before the file can be written')
            else:
                time = self._time

        f = make_raw_weather_data_filename(
            outLoc,
            self._Name,
            time,
        )

        self.files = [f]
        return f

    def write(self):
        """
        By calling the abstract/modular netcdf writer
        (RAiDER.utilFcns.write2NETCDF4core), write the weather model data
        and refractivity to an NETCDF4 file that can be accessed by external programs.
        """
        # Generate the filename
        f = self._out_name

        attrs_dict = {
            'Conventions': 'CF-1.6',
            'datetime': datetime.datetime.strftime(self._time, '%Y_%m_%dT%H_%M_%S'),
            'date_created': datetime.datetime.now().strftime('%Y_%m_%dT%H_%M_%S'),
            'title': 'Weather model data and delay calculations',
            'model_name': self._Name,
        }

        dimension_dict = {
            'x': ('x', self._xs),
            'y': ('y', self._ys),
            'z': ('z', self._zs),
            'latitude': (('y', 'x'), self._lats),
            'longitude': (('y', 'x'), self._lons),
            'datetime_utc': self._time.replace(tzinfo=None),
        }

        dataset_dict = {
            't': (('z', 'y', 'x'), self._t.swapaxes(0, 2).swapaxes(1, 2)),
            'p': (('z', 'y', 'x'), self._p.swapaxes(0, 2).swapaxes(1, 2)),
            'e': (('z', 'y', 'x'), self._e.swapaxes(0, 2).swapaxes(1, 2)),
            'wet': (('z', 'y', 'x'), self._wet_refractivity.swapaxes(0, 2).swapaxes(1, 2)),
            'hydro': (('z', 'y', 'x'), self._hydrostatic_refractivity.swapaxes(0, 2).swapaxes(1, 2)),
            'wet_total': (('z', 'y', 'x'), self._wet_ztd.swapaxes(0, 2).swapaxes(1, 2)),
            'hydro_total': (('z', 'y', 'x'), self._hydrostatic_ztd.swapaxes(0, 2).swapaxes(1, 2)),
        }

        ds = xarray.Dataset(data_vars=dataset_dict, coords=dimension_dict, attrs=attrs_dict)

        # Define units
        ds['t'].attrs['units'] = 'K'
        ds['e'].attrs['units'] = 'Pa'
        ds['p'].attrs['units'] = 'Pa'
        ds['wet'].attrs['units'] = 'dimentionless'
        ds['hydro'].attrs['units'] = 'dimentionless'
        ds['wet_total'].attrs['units'] = 'm'
        ds['hydro_total'].attrs['units'] = 'm'

        # Define standard names
        ds['t'].attrs['standard_name'] = 'temperature'
        ds['e'].attrs['standard_name'] = 'humidity'
        ds['p'].attrs['standard_name'] = 'pressure'
        ds['wet'].attrs['standard_name'] = 'wet_refractivity'
        ds['hydro'].attrs['standard_name'] = 'hydrostatic_refractivity'
        ds['wet_total'].attrs['standard_name'] = 'total_wet_refractivity'
        ds['hydro_total'].attrs['standard_name'] = 'total_hydrostatic_refractivity'

        # projection information
        ds['proj'] = 0
        for k, v in self._proj.to_cf().items():
            ds.proj.attrs[k] = v
        for var in ds.data_vars:
            ds[var].attrs['grid_mapping'] = 'proj'

        # write to file and return the filename
        ds.to_netcdf(f)
        return f


def make_weather_model_filename(name, time, ll_bounds) -> str:
    s = np.floor(ll_bounds[0])
    S = f'{np.abs(s):.0f}S' if s < 0 else f'{s:.0f}N'

    n = np.ceil(ll_bounds[1])
    N = f'{np.abs(n):.0f}S' if n < 0 else f'{n:.0f}N'

    w = np.floor(ll_bounds[2])
    W = f'{np.abs(w):.0f}W' if w < 0 else f'{w:.0f}E'

    e = np.ceil(ll_bounds[3])
    E = f'{np.abs(e):.0f}W' if e < 0 else f'{e:.0f}E'
    return f'{name}_{time.strftime("%Y_%m_%d_T%H_%M_%S")}_{S}_{N}_{W}_{E}.nc'


def make_raw_weather_data_filename(outLoc, name, time):
    """Filename generator for the raw downloaded weather model data."""
    date_string = datetime.datetime.strftime(time, '%Y_%m_%d_T%H_%M_%S')
    f = os.path.join(outLoc, f'{name}_{date_string}.nc')
    return f


def find_svp(t):
    """Calculate standard vapor presure. Should be model-specific."""
    # From TRAIN:
    # Could not find the wrf used equation as they appear to be
    # mixed with latent heat etc. Istead I used the equations used
    # in ERA-I (see IFS documentation part 2: Data assimilation
    # (CY25R1)). Calculate saturated water vapour pressure (svp) for
    # water (svpw) using Buck 1881 and for ice (swpi) from Alduchow
    # and Eskridge (1996) euation AERKi

    # TODO: figure out the sources of all these magic numbers and move
    # them somewhere more visible.
    # TODO: (Jeremy) - Need to fix/get the equation for the other
    # weather model types. Right now this will be used for all models,
    # except WRF, which is yet to be implemented in my new structure.
    t1 = 273.15  # O Celsius
    t2 = 250.15  # -23 Celsius

    tref = t - t1
    wgt = (t - t2) / (t1 - t2)
    svpw = 6.1121 * np.exp((17.502 * tref) / (240.97 + tref))
    svpi = 6.1121 * np.exp((22.587 * tref) / (273.86 + tref))

    svp = svpi + (svpw - svpi) * wgt**2
    ix_bound1 = t > t1
    svp[ix_bound1] = svpw[ix_bound1]
    ix_bound2 = t < t2
    svp[ix_bound2] = svpi[ix_bound2]

    svp = svp * 100
    return svp.astype(np.float32)


def get_mapping(proj):
    """Get CF-complient projection information from a proj."""
    # In case of WGS-84 lat/lon, keep it simple
    if proj.to_epsg() == 4326:
        return 'WGS84'
    else:
        return proj.to_wkt()


def checkContainment_raw(path_wm_raw, ll_bounds, buffer_deg: float = 1e-5) -> bool:
    """ "
    Checks if existing raw weather model contains
    requested ll_bounds.

    Args:
    ----------
    path_wm_raw : path to downloaded, uncropped weather model file
    ll_bounds: an array of floats (SNWE) demarcating bbox of targets
    buffer_deg : float
        For x-translates for extents that lie outside of world bounding box,
        this ensures that translates have some overlap. The default is 1e-5
        or ~11.1 meters.

    Returns:
    -------
    bool
        True if weather model contains bounding box of OutLats and outLons
        and False otherwise.
    """
    import xarray as xr

    ymin_input, ymax_input, xmin_input, xmax_input = ll_bounds
    input_box = box(xmin_input, ymin_input, xmax_input, ymax_input)

    with xr.open_dataset(path_wm_raw) as ds:
        try:
            ymin, ymax = ds.latitude.min(), ds.latitude.max()
            xmin, xmax = ds.longitude.min(), ds.longitude.max()
        except:
            ymin, ymax = ds.y.min(), ds.y.max()
            xmin, xmax = ds.x.min(), ds.x.max()

        xmin, xmax = np.mod(np.array([xmin, xmax]) + 180, 360) - 180
        weather_model_box = box(xmin, ymin, xmax, ymax)

    world_box = box(-180, -90, 180, 90)

    # Logger
    input_box_str = [f'{x:1.2f}' for x in [xmin_input, ymin_input, xmax_input, ymax_input]]
    weath_box_str = [f'{x:1.2f}' for x in [xmin, ymin, xmax, ymax]]

    weath_box_str = ', '.join(weath_box_str)
    input_box_str = ', '.join(input_box_str)

    # If the bounding box goes beyond the normal world extents
    # Look at two x-translates, buffer them, and take their union.
    if not world_box.contains(weather_model_box):
        logger.info('Considering x-translates of weather model +/-360 as bounding box outside of -180, -90, 180, 90')
        translates = [
            weather_model_box.buffer(buffer_deg),
            translate(weather_model_box, xoff=360).buffer(buffer_deg),
            translate(weather_model_box, xoff=-360).buffer(buffer_deg),
        ]
        weather_model_box = unary_union(translates)

    return weather_model_box.contains(input_box)
