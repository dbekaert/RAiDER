import datetime

import numpy as np
from pyproj import CRS

from RAiDER.mathFcns import round_date
from RAiDER.models.weatherModel import WeatherModel


class HRES(WeatherModel):
    '''
    Implement ECMWF models
    '''

    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        # model constants
        self._k1 = 0.776   # [K/Pa]
        self._k2 = 0.233   # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # 9 km horizontal grid spacing. This is only used for extending the download-buffer, i.e. not in subsequent processing.
        self._lon_res = 9. / 111
        self._lat_res = 9. / 111
        self._x_res = 9. / 111
        self._y_res = 9. / 111

        self._humidityType = 'q'
        # Default, pressure levels are 'pl'
        self._model_level_type = 'ml'
        self._expver = '1'
        self._classname = 'od'
        self._dataset = 'hres'
        self._Name = 'HRES'

        # Tuple of min/max years where data is available.
        self._valid_range = (datetime.datetime(1983, 4, 20), "Present")
        # Availability lag time in days
        self._lag_time = datetime.timedelta(hours=6)

        self._a = [
            0.000000, 2.000365, 3.102241, 4.666084, 6.827977,
            9.746966, 13.605424, 18.608931, 24.985718, 32.985710,
            42.879242, 54.955463, 69.520576, 86.895882, 107.415741,
            131.425507, 159.279404, 191.338562, 227.968948, 269.539581,
            316.420746, 368.982361, 427.592499, 492.616028, 564.413452,
            643.339905, 729.744141, 823.967834, 926.344910, 1037.201172,
            1156.853638, 1285.610352, 1423.770142, 1571.622925, 1729.448975,
            1897.519287, 2076.095947, 2265.431641, 2465.770508, 2677.348145,
            2900.391357, 3135.119385, 3381.743652, 3640.468262, 3911.490479,
            4194.930664, 4490.817383, 4799.149414, 5119.895020, 5452.990723,
            5798.344727, 6156.074219, 6526.946777, 6911.870605, 7311.869141,
            7727.412109, 8159.354004, 8608.525391, 9076.400391, 9562.682617,
            10065.978516, 10584.631836, 11116.662109, 11660.067383, 12211.547852,
            12766.873047, 13324.668945, 13881.331055, 14432.139648, 14975.615234,
            15508.256836, 16026.115234, 16527.322266, 17008.789063, 17467.613281,
            17901.621094, 18308.433594, 18685.718750, 19031.289063, 19343.511719,
            19620.042969, 19859.390625, 20059.931641, 20219.664063, 20337.863281,
            20412.308594, 20442.078125, 20425.718750, 20361.816406, 20249.511719,
            20087.085938, 19874.025391, 19608.572266, 19290.226563, 18917.460938,
            18489.707031, 18006.925781, 17471.839844, 16888.687500, 16262.046875,
            15596.695313, 14898.453125, 14173.324219, 13427.769531, 12668.257813,
            11901.339844, 11133.304688, 10370.175781, 9617.515625, 8880.453125,
            8163.375000, 7470.343750, 6804.421875, 6168.531250, 5564.382813,
            4993.796875, 4457.375000, 3955.960938, 3489.234375, 3057.265625,
            2659.140625, 2294.242188, 1961.500000, 1659.476563, 1387.546875,
            1143.250000, 926.507813, 734.992188, 568.062500, 424.414063,
            302.476563, 202.484375, 122.101563, 62.781250, 22.835938,
            3.757813, 0.000000, 0.000000
        ]

        self._b = [
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
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
            0.988500, 0.991984, 0.995003, 0.997630, 1.000000
        ]
    
    def update_a_b(self):
        # Before 2013-06-26, there were only 91 model levels. The mapping coefficients below are extracted based on https://www.ecmwf.int/en/forecasts/documentation-and-support/91-model-levels
        self._a = [0.000000,2.000040,3.980832,7.387186,12.908319,21.413612,33.952858,
                   51.746601,76.167656,108.715561,150.986023,204.637451,271.356506,
                   352.824493,450.685791,566.519226,701.813354,857.945801,1036.166504,
                   1237.585449,1463.163940,1713.709595,1989.874390,2292.155518,2620.898438,
                   2976.302246,3358.425781,3767.196045,4202.416504,4663.776367,5150.859863,
                   5663.156250,6199.839355,6759.727051,7341.469727,7942.926270,8564.624023,
                   9208.305664,9873.560547,10558.881836,11262.484375,11982.662109,12713.897461,
                   13453.225586,14192.009766,14922.685547,15638.053711,16329.560547,16990.623047,
                   17613.281250,18191.029297,18716.968750,19184.544922,19587.513672,19919.796875,
                   20175.394531,20348.916016,20434.158203,20426.218750,20319.011719,20107.031250,
                   19785.357422,19348.775391,18798.822266,18141.296875,17385.595703,16544.585938,
                   15633.566406,14665.645508,13653.219727,12608.383789,11543.166992,10471.310547,
                   9405.222656,8356.252930,7335.164551,6353.920898,5422.802734,4550.215820,3743.464355,
                   3010.146973,2356.202637,1784.854614,1297.656128,895.193542,576.314148,336.772369,
                   162.043427,54.208336,6.575628,0.003160,0.000000]
            
        self._b = [0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                   0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                   0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                   0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000014,
                   0.000055,0.000131,0.000279,0.000548,0.001000,0.001701,0.002765,0.004267,0.006322,
                   0.009035,0.012508,0.016860,0.022189,0.028610,0.036227,0.045146,0.055474,0.067316,
                   0.080777,0.095964,0.112979,0.131935,0.152934,0.176091,0.201520,0.229315,0.259554,
                   0.291993,0.326329,0.362203,0.399205,0.436906,0.475016,0.513280,0.551458,0.589317,
                   0.626559,0.662934,0.698224,0.732224,0.764679,0.795385,0.824185,0.850950,0.875518,
                   0.897767,0.917651,0.935157,0.950274,0.963007,0.973466,0.982238,0.989153,0.994204,
                   0.997630,1.000000]

    def load_weather(self, filename):
        '''
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        '''

        if filename is None:
            filename = self._files

        # read data from grib file
        lats, lons, xs, ys, t, q, lnsp, z = self._makeDataCubes(filename, verbose=False)

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z = z[::-1]
            lnsp = lnsp[::-1]
            t = t[:, ::-1]
            q = q[:, ::-1]
            lats = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z = z[..., ::-1]
            lnsp = lnsp[..., ::-1]
            t = t[..., ::-1]
            q = q[..., ::-1]
            lons = lons[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360
        self._proj = CRS.from_epsg(4326)

        self._t = t
        self._q = q
        geo_hgt, pres, hgt = self._calculategeoh(z, lnsp)

        # re-assign lons, lats to match heights
        _lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :], hgt.shape)
        _lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis], hgt.shape)
        # ys is latitude
        self._get_heights(_lats, hgt)
        h = self._zs.copy()

        # We want to support both pressure levels and true pressure grids.
        # If the shape has one dimension, we'll scale it up to act as a
        # grid, otherwise we'll leave it alone.
        if len(pres.shape) == 1:
            p = np.broadcast_to(pres[:, np.newaxis, np.newaxis], self._zs.shape)
        else:
            p = pres

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        p = np.transpose(p)
        t = np.transpose(t)
        q = np.transpose(q)
        h = np.transpose(h)
        _lats = np.transpose(_lats)
        _lons = np.transpose(_lons)

        # check this
        # data cube format should be lats,lons,heights
        p = p.swapaxes(0, 1)
        q = q.swapaxes(0, 1)
        t = t.swapaxes(0, 1)
        h = h.swapaxes(0, 1)
        _lats = _lats.swapaxes(0, 1)
        _lons = _lons.swapaxes(0, 1)

        # Flip all the axis so that zs are in order from bottom to top
        p = np.flip(p, axis=2)
        t = np.flip(t, axis=2)
        q = np.flip(q, axis=2)
        h = np.flip(h, axis=2)
        _lats = np.flip(_lats, axis=2)
        _lons = np.flip(_lons, axis=2)

        self._p = p
        self._q = q
        self._t = t
        self._lats = _lats
        self._lons = _lons
        self._xs = _lons.copy()
        self._ys = _lats.copy()
        self._zs = h

    def _makeDataCubes(self, fname, verbose=False):
        '''
        Create a cube of data representing temperature and relative humidity
        at specified pressure levels
        '''
        from scipy.io import netcdf as nc
        with nc.netcdf_file(fname, 'r', maskandscale=True) as f:
            # 0,0 to get first time and first level
            z = f.variables['z'][0][0].copy()
            lnsp = f.variables['lnsp'][0][0].copy()
            t = f.variables['t'][0].copy()
            q = f.variables['q'][0].copy()
            lats = f.variables['latitude'][:].copy()
            lons = f.variables['longitude'][:].copy()
            self._levels = f.variables['level'][:].copy()
            xs = lons.copy()
            ys = lats.copy()

        return lats, lons, xs, ys, t, q, lnsp, z

    def _fetch(self, lats, lons, time, out, Nextra=2):
        '''
        Fetch a weather model from ECMWF
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)

        # execute the search at ECMWF
        self._download_ecmwf_file(lat_min, lat_max, self._lat_res, lon_min, lon_max, self._lon_res, time, out)

    def _download_ecmwf_file(self, lat_min, lat_max, lat_step, lon_min, lon_max, lon_step, time, out):
        from ecmwfapi import ECMWFService

        server = ECMWFService("mars")

        corrected_date = round_date(time, datetime.timedelta(hours=6))

        if (time < datetime.datetime(2013, 6, 26, 0, 0, 0)):
            levels = 91
            self.update_a_b
        else:
            levels = 137
                
        server.execute({
           'class': self._classname,
           'date': datetime.datetime.strftime(corrected_date, "%Y-%m-%d"),
           'expver': "{}".format(self._expver),
           'levelist': "1/to/{0}".format(levels),
           'levtype': "ml",
           'param': "129/130/133/152",
           'stream': "oper",
           'time': "00:00:00",
           'type': "an",
           'step': "0",
           'grid': "{}/{}".format(lon_step, lat_step),
           'area': "{}/{}/{}/{}".format(lat_max, lon_min, lat_min, lon_max),
           'format': "netcdf", },
           out)
