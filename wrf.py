import numpy as np
import scipy.io.netcdf as netcdf
import pyproj

from weatherModel import WeatherModel


class WRF(WeatherModel):
    '''
    WRF class definition, based on the WeatherModel base class. 
    '''
    #TODO: finish implementing
    def __init__(self):
        WeatherModel.__init__(self)
        
        self._k1 = 0.776 # K/Pa
        self._k2 = 0.233 # K/Pa
        self._k3 = 3.75e3# K^2/Pa
    
        # Currently WRF is using RH instead of Q to get E
        self._humidityType = 'rh'


#    lla = pyproj.Proj(proj='latlong')
#
#    xs, ys = pyproj.transform(lla, projection, lons.flatten(), lats.flatten())
#    xs = xs.reshape(lats.shape)
#    ys = ys.reshape(lons.shape)
#
#    # At this point, if all has gone well, xs has every column the same,
#    # and ys has every row the same. Maybe they're not exactly the same
#    # (due to rounding errors), so we'll average them.
#    xs = np.mean(xs, axis=0)
#    ys = np.mean(ys, axis=1)

    def load_weather(self, file1, file2):
        '''
        Consistent class method to be implemented across all weather model types
        '''
        try:
            self._get_wm_nodes(file1)
            self._read_netcdf(file2)
        except KeyError:
            self._get_wm_nodes(file2)
            self._read_netcdf(file1)
            
        self._find_e_from_rh()
        self._get_wet_refractivity()
        self._get_hydro_refractivity() 
        
        # adjust the grid based on the height data
        self._adjust_grid()


    def _get_wm_nodes(self, nodeFile):
        with netcdf.netcdf_file(nodeFile, 'r', maskandscale=True) as outf:
            lats = outf.variables['XLAT'][0].copy() # Takes only the first date!
            lons = outf.variables['XLONG'][0].copy()

        lons[lons > 180] -= 360
        self._ys = lats
        self._xs = lons

    def _read_netcdf(self, weatherFile, defNul = None):
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
            checkUnits(humid.units.decode('utf-8'),'relative humidity') 
            checkUnits(geohvar.units.decode('utf-8'), 'geopotential') 

            # _FillValue is not always set, but when it is we want to read it
            tNull = getNullValue(temp)
            hNull = getNullValue(humid)
            gNull = getNullValue(geohvar)
            pNull = getNullValue(spvar)

            sp = spvar[0].copy()
            temps= temp[0].copy()
            humids= humid[0].copy()
            geoh= geohvar[0].copy()

        # Projection
        # See http://www.pkrc.net/wrf-lambert.html
        self._proj = pyproj.Proj(proj='lcc', lat_1=lat1,
                             lat_2=lat2, lat_0=lat0,
                             lon_0=lon0, a=6370, b=6370,
                             towgs84=(0,0,0), no_defs=True)

        temps[temps==tNull] = np.nan
        sp[sp==pNull] = np.nan
        humids[humids==hNull] = np.nan
        geoh[geoh==gNull] = np.nan

        self._t = temps
        self._rh= humids
        self._zs= geoh

        if len(sp.shape) == 1:
            self._p = np.broadcast_to(sp[:, np.newaxis, np.newaxis],
                                        self._zs.shape)
        else:
            self._p = sp 
    

class UnitTypeError(Exception):
    '''
    Define a unit type exception for easily formatting
    error messages for units
    '''
    def __init___(self,varName, unittype):
        msg = "Unknown units for {}: '{}'".format(varName, unittype)
        Exception.__init__(self,msg)


def checkUnits(unitCheck, varName):
    '''
    Implement a check that the units are as expected
    '''
    unitDict = {'pressure': 'Pa', 'temperature':'K', 'relative humidity': '%', 'geopotential': 'm'}
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
    

