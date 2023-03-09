import numpy as np
import scipy.io.netcdf as netcdf
from pyproj import CRS, Transformer

from RAiDER.models.weatherModel import WeatherModel, TIME_RES


# Need to incorporate this snippet into this part of the code.
# was formally in delay.py
# if weather_model_name == 'wrf':
#    # Let lats and lons to weather model nodes if necessary
#    #TODO: Need to fix the case where lats are None, because
#    # the weather model need not be in latlong projection
#    if lats is None:
#        lats, lons = wrf.wm_nodes(*weather_files)
#
class WRF(WeatherModel):
    '''
    WRF class definition, based on the WeatherModel base class.
    '''
    # TODO: finish implementing

    def __init__(self):
        WeatherModel.__init__(self)

        self._k1 = 0.776  # K/Pa
        self._k2 = 0.233  # K/Pa
        self._k3 = 3.75e3  # K^2/Pa


        # Currently WRF is using RH instead of Q to get E
        self._humidityType = 'rh'
        self._Name = 'WRF'
        self._time_res = TIME_RES[self._Name]

    def _fetch(self):
        pass

    def load_weather(self, file1, file2, *args, **kwargs):
        '''
        Consistent class method to be implemented across all weather model types
        '''
        try:
            lons, lats = self._get_wm_nodes(file1)
            self._read_netcdf(file2)
        except KeyError:
            self._get_wm_nodes(file2)
            self._read_netcdf(file1)

        # WRF doesn't give us the coordinates of the points in the native projection,
        # only the coordinates in lat/long. Ray transformed these to the native
        # projection, then used an average to enforce a regular grid. It does matter
        # for the interpolation whether the grid is regular.
        lla = CRS.from_epsg(4326)
        t = Transformer.from_proj(lla, self._proj)
        xs, ys = t.transform(lons.flatten(), lats.flatten())
        xs = xs.reshape(lons.shape)
        ys = ys.reshape(lats.shape)

        # Expected accuracy here is to two decimal places (five significant digits)
        xs = np.mean(xs, axis=0)
        ys = np.mean(ys, axis=1)

        _xs = np.broadcast_to(xs[np.newaxis, np.newaxis, :],
                              self._p.shape)
        _ys = np.broadcast_to(ys[np.newaxis, :, np.newaxis],
                              self._p.shape)
        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        self._p = np.transpose(self._p)
        self._t = np.transpose(self._t)
        self._rh = np.transpose(self._rh)
        self._ys = np.transpose(_ys)
        self._xs = np.transpose(_xs)
        self._zs = np.transpose(self._zs)

        # TODO: Not sure if WRF provides this
        self._levels = list(range(self._zs.shape[2]))

    def _get_wm_nodes(self, nodeFile):
        with netcdf.netcdf_file(nodeFile, 'r', maskandscale=True) as outf:
            lats = outf.variables['XLAT'][0].copy()  # Takes only the first date!
            lons = outf.variables['XLONG'][0].copy()

        lons[lons > 180] -= 360

        return lons, lats

    def _read_netcdf(self, weatherFile, defNul=None):
        """
        Read weather variables from a netCDF file
        """
        if defNul is None:
            defNul = np.nan

        # TODO: it'd be cool to use some kind of units package
        # TODO: extract partial pressure directly (q?)
        with netcdf.netcdf_file(weatherFile, 'r', maskandscale=True) as f:
            spvar = f.variables['P_PL']
            temp = f.variables['T_PL']
            humid = f.variables['RH_PL']
            geohvar = f.variables['GHT_PL']

            lon0 = f.STAND_LON.copy()
            lat0 = f.MOAD_CEN_LAT.copy()
            lat1 = f.TRUELAT1.copy()
            lat2 = f.TRUELAT2.copy()

            checkUnits(spvar.units.decode('utf-8'), 'pressure')
            checkUnits(temp.units.decode('utf-8'), 'temperature')
            checkUnits(humid.units.decode('utf-8'), 'relative humidity')
            checkUnits(geohvar.units.decode('utf-8'), 'geopotential')

            # _FillValue is not always set, but when it is we want to read it
            tNull = getNullValue(temp)
            hNull = getNullValue(humid)
            gNull = getNullValue(geohvar)
            pNull = getNullValue(spvar)

            sp = spvar[0].copy()
            temps = temp[0].copy()
            humids = humid[0].copy()
            geoh = geohvar[0].copy()

            spvar = None
            temp = None
            humid = None
            geohvar = None

        # Projection
        # See http://www.pkrc.net/wrf-lambert.html
        earthRadius = 6370e3  # <- note Ray had a bug here
        p1 = CRS(proj='lcc', lat_1=lat1,
                 lat_2=lat2, lat_0=lat0,
                 lon_0=lon0, a=earthRadius, b=earthRadius,
                 towgs84=(0, 0, 0), no_defs=True)
        self._proj = p1

        temps[temps == tNull] = np.nan
        sp[sp == pNull] = np.nan
        humids[humids == hNull] = np.nan
        geoh[geoh == gNull] = np.nan

        self._t = temps
        self._rh = humids

        # Zs are problematic because any z below the topography is nan.
        # For a temporary fix, I will assign any nan value to equal the
        # nanmean of that level.
        zmeans = np.nanmean(geoh, axis=(1, 2))
        nz, ny, nx = geoh.shape
        Zmeans = np.tile(zmeans, (nx, ny, 1))
        Zmeans = Zmeans.T
        ix = np.isnan(geoh)
        geoh[ix] = Zmeans[ix]
        self._zs = geoh

        if len(sp.shape) == 1:
            self._p = np.broadcast_to(
                sp[:, np.newaxis, np.newaxis], self._zs.shape)
        else:
            self._p = sp


class UnitTypeError(Exception):
    '''
    Define a unit type exception for easily formatting
    error messages for units
    '''

    def __init___(self, varName, unittype):
        msg = "Unknown units for {}: '{}'".format(varName, unittype)
        Exception.__init__(self, msg)


def checkUnits(unitCheck, varName):
    '''
    Implement a check that the units are as expected
    '''
    unitDict = {'pressure': 'Pa', 'temperature': 'K', 'relative humidity': '%', 'geopotential': 'm'}
    if unitCheck != unitDict[varName]:
        raise UnitTypeError(varName, unitCheck)


def getNullValue(var):
    '''
    Get the null (or fill) value if it exists, otherwise set the null value to defNullValue
    '''
    # NetCDF files have the ability to record their nodata value, but in the
    # particular NetCDF files that I'm reading, this field is left
    # unspecified and a nodata value of -999 is used. The solution I'm using
    # is to check if nodata is specified, and otherwise assume it's -999.
    _default_fill_value = -999

    try:
        var_fill = var._FillValue
    except AttributeError:
        var_fill = _default_fill_value

    return var_fill
