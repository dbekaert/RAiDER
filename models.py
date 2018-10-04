
# standard imports
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import os

# local imports
import constants as const
import util
from util import robmin, robmax

class WeatherModel():
    '''
    Implement a generic weather model for getting estimated SAR delays
    '''
    def __init__(self):
        self._k1 = None
        self._k2 = None
        self._k3 = None
        self._humidityType = 'q'
        self._a = []
        self._b = []
        self._zmin = const._ZMIN
        self._zref = const._ZMAX
        self._xs = np.empty((1, 1, 1)) 
        self._ys = np.empty((1, 1, 1))
        self._zs = np.empty((1, 1, 1))
        self._lats = np.empty((1, 1, 1))
        self._lons = np.empty((1, 1, 1))
        self._heights = np.empty((1, 1, 1))
        self._lon_res = None
        self._lat_res = None
        self._llaproj = pyproj.Proj(proj='latlong')
        self._ecefproj = pyproj.Proj(proj='geocent')
        self._pure_scipy_interp = False
        self._p = None
        self._q = None
        self._t = None
        self._e = None
        self._classname = None 
        self._dataset = None
        self._wet_refractivity = None
        self._hydrostatic_refractivity = None
        self._svp = None
        self._R_v = 461.524
        self._R_d = 287.053

        
    def __repr__(self):
        string = '\n'
        string += '======Weather Model class object=====\n'
        string += 'Number of points in Lat/Lon = {}/{}\n'.format(*self._lats.shape[1:])
        string += 'Total number of grid points (3D): {}\n'.format(np.prod(self._xs.shape))
        string += 'Latitude resolution: {}\n'.format(self._lat_res)
        string += 'Longitude resolution: {}\n'.format(self._lon_res)
        string += 'ZMIN: {}\n'.format(self._zmin)
        string += 'ZMAX: {}\n'.format(self._zref)
        string += 'Minimum/Maximum latitude: {}/{}\n'\
                  .format(round(robmin(self._lats), 2), round(robmax(self._lats), 2))
        string += 'Minimum/Maximum longitude: {}/{}\n'\
                  .format(round(robmin(self._lons), 2), round(robmax(self._lons), 2))
        string += 'Minimum/Maximum heights: {}/{}\n'\
                  .format(robmin(self._heights), round(robmax(self._heights), 2))
        string += '=====================================\n'
        string += 'k1 = {}\n'.format(self._k1)
        string += 'k2 = {}\n'.format(self._k2)
        string += 'k3 = {}\n'.format(self._k3)
        string += 'Humidity type = {}\n'.format(self._humidityType)
        string += 'Use pure scipy interpolation: {}\n'.format(self._pure_scipy_interp)
        string += '=====================================\n'
        string += 'Class name: {}\n'.format(self._classname)
        string += 'Dataset: {}\n'.format(self._dataset)
        string += '=====================================\n'
        return str(string)


    def plot(self):
        try:
            fig = plt.scatter(self._lons, self._lats)
        except Exception as e:
            print(e)
        return fig

    def fetch(self, lats, lons, time, out):
        pass

    def load(self, filename):
        self.load_weather(filename)

    def loadInterp(self):
        return self._getInterpFcn()

    def setInterpMethod(self, use_pure_scipy = True):
        self._pure_scipy_interp = use_pure_scipy        

    def load_weather(self, filename):
        pass

    def _get_heights(self, geo_hgt, geo_ht_fill = np.nan):
        '''
        Transform geo heights to actual heights
        '''
        geo_ht_fix = np.where(geo_hgt!= geo_ht_fill, geo_hgt, np.nan)
        self._heights = util.geo_to_ht(self._lats, self._lons, geo_ht_fix)

    def _find_e_from_q(self):
        """Calculate e, partial pressure of water vapor."""
        self._find_svp()
        # We have q = w/(w + 1), so w = q/(1 - q)
        w = self._q/(1 - self._q)
        self._e = w*self._R_v*(self._p - self._svp)/self._R_d

    def _get_wet_refractivity(self):
        '''
        Calculate the wet delay from pressure, temperature, and e
        '''
        self._wet_refractivity = self._k2*self._e/self._t+ self._k3*self._e/self._t**2
        
    def _get_hydro_refractivity(self):
        '''
        Calculate the hydrostatic delay from pressure and temperature
        '''
        self._hydrostatic_refractivity = self._k1*self._p/self._t

    def _adjust_grid(self):
        if self._zmin < np.min(self._heights):
            # add in a new layer at zmin
            new_heights = np.zeros(self._heights.shape[1:]) + self._zmin
            self._heights = np.concatenate((new_heights[np.newaxis], self._heights))

            # need to extrapolate the other variables now
            new_pressures = util._least_nonzero(self._p)
            self._p= np.concatenate((new_pressures[np.newaxis], self._p))

            new_temps = util._least_nonzero(self._t)
            self._t= np.concatenate((new_temps[np.newaxis], self._t))

            new_humids = util._least_nonzero(self._q)
            self._q= np.concatenate((new_humids[np.newaxis], self._q))

    def _getInterpFcn(self):
        '''
        Get the interpolation fcn for interpolating new points
        '''
        from reader import LinearModel as lm
        interpFcn = lm(xs=self._xs, ys=self._ys, heights=self._heights,
                           pressure=self._p,
                           temperature=self._t, humidity=self._q,
                           k1=self._k1, k2=self._k2, k3=self._k3, projection=self._llaproj,
                           scipy_interpolate=self._pure_scipy_interp,
                           humidity_type=self._humidityType, zmin=self._zmin)
        return interpFcn


    def _find_svp(self):
        """Calculate standard vapor presure."""
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
        svpw = (6.1121
                * np.exp((17.502*(self._t- 273.16))/(240.97 + self._t- 273.16)))
        svpi = (6.1121
                * np.exp((22.587*( self._t - 273.16))/(273.86 + self._t  - 273.16)))
        tempbound1 = 273.16  # 0
        tempbound2 = 250.16  # -23
    
        svp = svpw
        wgt = (self._t - tempbound2)/(tempbound1 - tempbound2)
        svp = svpi + (svpw - svpi)*wgt**2
        ix_bound1 =self._t > tempbound1
        svp[ix_bound1] = svpw[ix_bound1]
        ix_bound2 =self._t < tempbound2
        svp[ix_bound2] = svpi[ix_bound2]
    
        self._svp = svp * 100


class ECMWF(WeatherModel):
    '''
    Implement ECMWF models
    '''
    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233 # [K/Pa]
        self._k3 = 3.75e3 # [K^2/Pa]

        # model resolution
        self._lon_res = 0.2
        self._lat_res = 0.2


    def load_weather(self, filename):
        '''
        Consistent class method to be implemented across all weather model types
        '''
        self._load_model_level(filename)

    def _load_model_level(self, fname):
        from scipy.io import netcdf as nc
        with nc.netcdf_file(fname, 'r', maskandscale=True) as f:
            # 0,0 to get first time and first level
            z = f.variables['z'][0][0].copy()
            lnsp = f.variables['lnsp'][0][0].copy()
            t = f.variables['t'][0].copy()
            q = f.variables['q'][0].copy()
            lats = f.variables['latitude'][:].copy()
            lons = f.variables['longitude'][:].copy()


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


        self._lons = lons
        self._lats = lats
        self._t = t
        self._q = q

        geo_hgt,pres = util.calculategeoh(self._a, self._b, z, lnsp, self._t, self._q)
        self._get_heights(geo_hgt)

        # We want to support both pressure levels and true pressure grids.
        # If the shape has one dimension, we'll scale it up to act as a
        # grid, otherwise we'll leave it alone.
        if len(pres.shape) == 1:
            self._p = np.broadcast_to(pres[:, np.newaxis, np.newaxis],
                                        self._heights.shape)
        else:
            self._p = pres

        # re-assign lons, lats to match heights
        self._lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :],
                                     self._heights.shape)
        self._lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis],
                                     self._heights.shape)

        # Also get the earth-centered coordinates
        self._xs, self._ys, self._zs = util.lla2ecef(self._lats.flatten(), 
                                                     self._lons.flatten(), 
                                                     self._heights.flatten())

#        # Replace the non-useful values by NaN
#        # TODO: replace np.where with dask.where
#        self.t = np.where(self.t!= tempNoData, self.t, np.nan)
#        self.q = np.where(self.q!= humidNoData, self.q, np.nan)
#
    
        # We've got to recover the grid of lat, lon
        # TODO: replace np.meshgrid with dask.meshgrid
        #self._get_ll()

        # compute e, wet delay, and hydrostatic delay
        self._find_e_from_q()
        self._get_wet_refractivity()
        self._get_hydro_refractivity() 
        
        # adjust the grid based on the height data
        self._adjust_grid()



    def fetch(self, lats, lons, time, out, Nextra = 2):
        '''
        Fetch a weather model from ECMWF
        '''
        lat_min = np.nanmin(lats) - Nextra*self._lat_res
        lat_max = np.nanmax(lats) + Nextra*self._lat_res
        lon_min = np.nanmin(lons) - Nextra*self._lon_res
        lon_max = np.nanmax(lons) + Nextra*self._lon_res
 
        self._get_from_ecmwf(
                lat_min, lat_max, self._lat_res, lon_min, lon_max, self._lon_res, time,
                out)

    def _get_from_ecmwf(self, lat_min, lat_max, lat_step, lon_min, lon_max,
                       lon_step, time, out):
        import ecmwfapi

        server = ecmwfapi.ECMWFDataServer()

        corrected_date = util.round_date(time, datetime.timedelta(hours=6))

        server.retrieve({
            "class": self._classname,  # ERA-Interim
            'dataset': self._dataset,
            "expver": "1",
            # They warn me against all, but it works well
            "levelist": 'all',
            "levtype": "ml",  # Model levels
            "param": "lnsp/q/z/t",  # Necessary variables
            "stream": "oper",
            # date: Specify a single date as "2015-08-01" or a period as
            # "2015-08-01/to/2015-08-31".
            "date": datetime.datetime.strftime(corrected_date, "%Y-%m-%d"),
            # type: Use an (analysis) unless you have a particular reason to
            # use fc (forecast).
            "type": "an",
            # time: With type=an, time can be any of
            # "00:00:00/06:00:00/12:00:00/18:00:00".  With type=fc, time can
            # be any of "00:00:00/12:00:00",
            "time": datetime.datetime.strftime(corrected_date, "%H:%M:%S"),
            # step: With type=an, step is always "0". With type=fc, step can
            # be any of "3/6/9/12".
            "step": "0",
            # grid: Only regular lat/lon grids are supported.
            "grid": f'{lat_step}/{lon_step}',
            "area": f'{lat_max}/{lon_min}/{lat_min}/{lon_max}',    # area: N/W/S/E
            "format": "netcdf",
            "resol": "av",
            "target": out,    # target: the name of the output file.
        })


class ERAI(ECMWF):
    # A and B parameters to calculate pressures for model levels,
    #  extracted from an ECMWF ERA-Interim GRIB file and then hardcoded here
    def __init__(self):
        ECMWF.__init__(self)
        self._a = [0.0000000000e+000, 2.0000000000e+001, 3.8425338745e+001,
             6.3647796631e+001, 9.5636962891e+001, 1.3448330688e+002,
             1.8058435059e+002, 2.3477905273e+002, 2.9849584961e+002,
             3.7397192383e+002, 4.6461816406e+002, 5.7565112305e+002,
             7.1321801758e+002, 8.8366040039e+002, 1.0948347168e+003,
             1.3564746094e+003, 1.6806403809e+003, 2.0822739258e+003,
             2.5798886719e+003, 3.1964216309e+003, 3.9602915039e+003,
             4.9067070313e+003, 6.0180195313e+003, 7.3066328125e+003,
             8.7650546875e+003, 1.0376125000e+004, 1.2077445313e+004,
             1.3775324219e+004, 1.5379804688e+004, 1.6819472656e+004,
             1.045183594e+004, 1.9027695313e+004, 1.9755109375e+004,
             2.0222203125e+004, 2.0429863281e+004, 2.0384480469e+004,
             2.0097402344e+004, 1.9584328125e+004, 1.8864750000e+004,
             1.7961359375e+004, 1.6899468750e+004, 1.5706449219e+004,
             1.4411125000e+004, 1.3043218750e+004, 1.1632757813e+004,
             1.0209500000e+004, 8.8023554688e+003, 7.4388046875e+003,
             6.1443164063e+003, 4.9417773438e+003, 3.8509133301e+003,
             2.8876965332e+003, 2.0637797852e+003, 1.3859125977e+003,
             8.5536181641e+002, 4.6733349609e+002, 2.1039389038e+002,
             6.5889236450e+001, 7.3677425385e+000, 0.0000000000e+000,
             0.0000000000e+000]
        self._b = [0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
             0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
             0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
             0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
             0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
             0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
             0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
             0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
             7.5823496445e-005, 4.6139489859e-004, 1.8151560798e-003,
             5.0811171532e-003, 1.1142909527e-002, 2.0677875727e-002,
             3.4121163189e-002, 5.1690407097e-002, 7.3533833027e-002,
             9.9674701691e-002, 1.3002252579e-001, 1.6438430548e-001,
             2.0247590542e-001, 2.4393314123e-001, 2.8832298517e-001,
             3.3515489101e-001, 3.8389211893e-001, 4.3396294117e-001,
             4.8477154970e-001, 5.3570991755e-001, 5.8616840839e-001,
             6.3554745913e-001, 6.8326860666e-001, 7.2878581285e-001,
             7.7159661055e-001, 8.1125342846e-001, 8.4737491608e-001,
             8.7965691090e-001, 9.0788388252e-001, 9.3194031715e-001,
             9.5182150602e-001, 9.6764522791e-001, 9.7966271639e-001,
             9.8827010393e-001, 9.9401944876e-001, 9.9763011932e-001,
             1.0000000000e+000]
    
        self._classname = 'ei'
        self._dataset = 'interim'


class ERA5(ECMWF):
    # I took this from
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels.
    def __init__(self):
        ECMWF.__init__(self)
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
    
        self._classname = 'ea'
        self._dataset = 'era5'


class MERRA2(WeatherModel):
    """Reader for MERRA-2 model.
    
    Only supports pressure levels right now, that may change someday.
    """
    import urllib.request
    import json

    _k1 = None
    _k2 = None
    _k3 = None
  
    def fetch(self, lats, lons, time, out):
        """Fetch MERRA-2."""
        # TODO: This function doesn't work right now. I'm getting a 302
        # Found response, with a message The document has moved. That's
        # annoying, it'd be nice if pydap would just follow to the new
        # page. I don't have time to debug this now, so I'll just drop
        # this comment here.
        import pydap.client
        import pydap.cas.urs
        import json
        from scipy.io import netcdf as nc

        lat_min = int(np.min(lats))
        lat_max = int(np.max(lats) + 0.5)
        lon_min = int(np.min(lons))
        lon_max = int(np.max(lons) + 0.5)
        url = ('https://goldsmr5.gesdisc.eosdis.nasa.gov:443/'\
               'opendap/MERRA2/M2I6NPANA.5.12.4/1980/01/'\
               'MERRA2_100.inst6_3d_ana_Np.19800101.nc4')
        config = os.path.expandvars(os.path.join('$HOME', '.urs-auth'))
        with open(config, 'r') as f:
            j = json.load(f)
            username = j['username']
            password = j['password']

        session = pydap.cas.urs.setup_session(
            username, password, check_url=url)
        # TODO: probably use a context manager so it gets closed
        dataset = pydap.client.open_url(url, session=session)

        # TODO: gotta figure out how to index as time
        timeNum = 0

        with nc.netcdf_file(out, 'w') as f:
            def index(ds):
                return ds[timeNum][:][lat_min:lat_max][lon_min:lon_max]
            t = f.createVariable('T', float, [])
            t[:] = index(dataset['T'])
            lats = f.createVariable('lat', float, [])
            lats[:] = dataset['lat'][lat_min:lat_max]
            lons = f.createVariable('lon', float, [])
            lons[:] = dataset['lon'][lon_min:lon_max]
            q = f.createVariable('QV', float, [])
            q[:] = index(dataset['QV'])
            z = f.createVariable('H', float, [])
            z[:] = index(dataset['H'])
            p = f.createVariable('lev', float, [])
            p[:] = dataset['lev'][:]

    def load_pressure_level(self, filename):
        from scipy.io import netcdf as nc
        with nc.netcdf_file(
                filename, 'r', maskandscale=True) as f:
            lats = f.variables['lat'][:].copy()
            lons = f.variables['lon'][:].copy()
            t = f.variables['T'][0].copy()
            q = f.variables['QV'][0].copy()
            z = f.variables['H'][0].copy()
            p = f.variables['lev'][0].copy()
        proj = pyproj.Proj('lla')
        return lats, lons, proj, t, q, z, p

    def weather_and_nodes(self, filename):
        return self.weather_and_nodes_from_pressure_levels(filename)

    def weather_and_nodes_from_pressure_levels(self, filename):
        pass

    def _url_builder(self, time, lat_min, lat_step, lat_max, lon_min, lon_step, lon_max):
        if lon_max < 0:
            lon_max += 360
        if lon_min < 0:
            lon_min += 360
        if lon_min > lon_max:
            lon_max, lon_min = lon_min, lon_max
        timeStr = '[0]'  # TODO: change
        latStr = '[{}:{}:{}]'.format(lat_min, lat_step, lat_max)
        lonStr = '[{}:{}:{}]'.format(lon_min,lon_step, lon_max)  # TODO: wrap
        lvs = '[0:1:41]'
        combined = '{}{}{}{}'.format(timeStr, lvs, latStr, lonStr)
        return ('https://goldsmr5.gesdisc.eosdis.nasa.gov:443/opendap/MERRA2/' +
                'M2I6NPANA.5.12.4/{}/'.format(time.strftime("%Y/%m")) + 
                'MERRA2_100.inst6_3d_ana_Np.{}.nc4.nc?'.format(time.strftime("%Y%m%d")) + 
                'T{},H{},lat{},lev{},lon{},'.format(combined, combined, latStr, lvs, lonStr) + 
                'time{}'.format(timeStr))

