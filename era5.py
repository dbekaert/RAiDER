import datetime 
import numpy as np
import pyproj

import util
from ecmwf import ECMWF

class ERA5(ECMWF):
    # I took this from
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels.
    def __init__(self):
        ECMWF.__init__(self)

        self._humidityType = 'rh'
        self._model_level_type = 'pl' # Default, pressure levels are 'pl'
        self._expver = '0001'
        self._classname = 'ea'
        self._dataset = 'era5'
        self._Name = 'ERA-5'

        self._valid_range = (datetime.date(1950,1,1),) # Tuple of min/max years where data is available. 
        self._lag_time = datetime.timedelta(days =30) # Availability lag time in days

        self._a = [0.000000,     2.000365,     3.102241,     4.666084,     6.827977,
             9.746966,     13.605424,    18.608931,    24.985718,    32.985710,
             42.879242,    54.955463,    69.520576,    86.895882,    107.415741,
             131.425507,   159.279404,   191.338562,   227.968948,   269.539581,
             316.420746,   368.982361,   427.592499,   492.616028,   564.413452,
             643.339905,   729.744141,   823.967834,   926.344910,   1037.201172,
             1156.853638,  1285.610352,  1423.770142,  1571.622925,  1729.448975,
             1897.519287,  2076.095947,  2265.431641,  2465.770508,  2677.348145,
             2900.391357,  3135.119385,  3381.743652,  3640.468262,  3911.490479,
             4194.930664,  4490.817383,  4799.149414,  5119.895020,  5452.990723,
             5798.344727,  6156.074219,  6526.946777,  6911.870605,  7311.869141,
             7727.412109,  8159.354004,  8608.525391,  9076.400391,  9562.682617,
             10065.978516, 10584.631836, 11116.662109, 11660.067383, 12211.547852,
             12766.873047, 13324.668945, 13881.331055, 14432.139648, 14975.615234,
             15508.256836, 16026.115234, 16527.322266, 17008.789063, 17467.613281,
             17901.621094, 18308.433594, 18685.718750, 19031.289063, 19343.511719,
             19620.042969, 19859.390625, 20059.931641, 20219.664063, 20337.863281,
             20412.308594, 20442.078125, 20425.718750, 20361.816406, 20249.511719,
             20087.085938, 19874.025391, 19608.572266, 19290.226563, 18917.460938,
             18489.707031, 18006.925781, 17471.839844, 16888.687500, 16262.046875,
             15596.695313, 14898.453125, 14173.324219, 13427.769531, 12668.257813,
             11901.339844, 11133.304688, 10370.175781, 9617.515625,  8880.453125,
             8163.375000,  7470.343750,  6804.421875,  6168.531250,  5564.382813,
             4993.796875,  4457.375000,  3955.960938,  3489.234375,  3057.265625,
             2659.140625,  2294.242188,  1961.500000,  1659.476563,  1387.546875,
             1143.250000,  926.507813,   734.992188,   568.062500,   424.414063,
             302.476563,   202.484375,   122.101563,   62.781250,    22.835938,
             3.757813,     0.000000,     0.000000]

        self._b = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
             0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
             0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
             0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
             0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
             0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
             0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
             0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000007,
             0.000024, 0.000059, 0.000112, 0.000199, 0.000340, 0.000562, 0.000890,
             0.001353, 0.001992, 0.002857, 0.003971, 0.005378, 0.007133, 0.009261,
             0.011806, 0.014816, 0.018318, 0.022355, 0.026964, 0.032176, 0.038026,
             0.044548, 0.051773, 0.059728, 0.068448, 0.077958, 0.088286, 0.099462,
             0.111505, 0.124448, 0.138313, 0.153125, 0.168910, 0.185689, 0.203491,
             0.222333, 0.242244, 0.263242, 0.285354, 0.308598, 0.332939, 0.358254,
             0.384363, 0.411125, 0.438391, 0.466003, 0.493800, 0.521619, 0.549301,
             0.576692, 0.603648, 0.630036, 0.655736, 0.680643, 0.704669, 0.727739,
             0.749797, 0.770798, 0.790717, 0.809536, 0.827256, 0.843881, 0.859432,
             0.873929, 0.887408, 0.899900, 0.911448, 0.922096, 0.931881, 0.940860,
             0.949064, 0.956550, 0.963352, 0.969513, 0.975078, 0.980072, 0.984542,
             0.988500, 0.991984, 0.995003, 0.997630, 1.000000]
    
    def fetch(self, lats, lons, time, out, Nextra = 2):
        '''
        Fetch a weather model from ECMWF
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)

        # execute the search at ECMWF
        self._get_from_cds(
                lat_min, lat_max, self._lat_res, lon_min, lon_max, self._lon_res, time,
                out)

    def load_weather(self, f):
        self._load_pressure_level(f)

    def _load_pressure_level(self, filename):
        from scipy.io import netcdf as nc
        with nc.netcdf_file(
                filename, 'r', maskandscale=True) as f:
            lats = f.variables['latitude'][:].copy()
            lons = f.variables['longitude'][:].copy()
            t = f.variables['t'][0].copy()
            q = f.variables['q'][0].copy()
            r = f.variables['r'][0].copy()
            z = f.variables['z'][0].copy()
            levels = f.variables['level'][:].copy()*100
            #TODO: note that levels is pressure
            #TODO: check ECMWF for variable ordering and test for consistency
            # may need to email ECMWF people for clarity

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z = z[::-1]
            t = t[:, ::-1]
            q = q[:, ::-1]
            r = r[:, ::-1]
            lats = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z = z[..., ::-1]
            t = t[..., ::-1]
            q = q[..., ::-1]
            r = r[..., ::-1]
            lons = lons[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360
        self._proj = pyproj.Proj(proj='latlong')

        self._t = t
        self._q = q
        self._rh = r

        geo_hgt = z/self._g0

        # re-assign lons, lats to match heights
        _lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :],
                                     geo_hgt.shape)
        _lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis],
                                     geo_hgt.shape)

        # correct heights for latitude
        self._get_heights(_lats, geo_hgt)

        self._p = np.broadcast_to(levels[:, np.newaxis, np.newaxis],
                                  self._zs.shape)

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        self._p = np.transpose(self._p)
        self._t = np.transpose(self._t)
        self._q = np.transpose(self._q)
        self._rh = np.transpose(self._rh)
        self._ys = np.transpose(_lats)
        self._xs = np.transpose(_lons)
        self._zs = np.transpose(self._zs)

        # check this
        self._xs = self._xs.swapaxes(0,1)
        self._ys = self._ys.swapaxes(0,1)
        self._zs = self._zs.swapaxes(0,1)
        self._rh = self._rh.swapaxes(0,1)
        self._p = self._p.swapaxes(0,1)
        self._q = self._q.swapaxes(0,1)
        self._t = self._t.swapaxes(0,1)

        # For some reason z is opposite the others
        self._p = np.flip(self._p, axis = 2)
        self._t = np.flip(self._t, axis = 2)
        self._q = np.flip(self._q, axis = 2)
        self._rh = np.flip(self._rh, axis = 2)

