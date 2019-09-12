import datetime
import numpy as np
import os
import pyproj
import re

from RAiDER.models.weatherModel import WeatherModel


def Model():
    return HRRR()

class HRRR(WeatherModel):
    # I took this from
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels.
    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        self._humidityType = 'q'
        self._model_level_type = 'pl' # Default, pressure levels are 'pl'
        self._expver = '0001'
        self._classname = 'hrrr'
        self._dataset = 'hrrr'

        self._valid_range = (datetime.date(2018,7,15),) # Tuple of min/max years where data is available. 
        self._lag_time = datetime.timedelta(hours=3) # Availability lag time in days

        # model constants: TODO: need to update/double-check these
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233 # [K/Pa]
        self._k3 = 3.75e3 # [K^2/Pa]

        # 3 km horizontal grid spacing
        self._lon_res = 3./111
        self._lat_res = 3./111

        self._Nproc = 1
        self._Name = 'HRRR'
        self._Npl = 0

        # Projection
        # See https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb and code lower down
        # '262.5:38.5:38.5:38.5 237.280472:1799:3000.00 21.138123:1059:3000.00'
        # 'lov:latin1:latin2:latd lon1:nx:dx lat1:ny:dy'
        # LCC parameters
        lon0 = 262.5
        lat0 = 38.5
        lat1 = 38.5
        lat2 = 38.5
        p1 = pyproj.Proj(proj='lcc', lat_1=lat1,
                             lat_2=lat2, lat_0=lat0,
                             lon_0=lon0, a=6370, b=6370,
                             towgs84=(0,0,0), no_defs=True)
        self._proj = p1

    def load(self, filename = None):
        if self._p is not None:
            pass 
        else:
            self.load_weather(filename = filename, download_only = False, nProc = self._Nproc)

    def fetch(self, lats, lons, time, out, Nextra = 2):
        '''
        Fetch weather model data from HRRR
        '''
        # bounding box plus a buffer
        bounds = self._get_ll_bounds(lats, lons, Nextra)

        # execute the search at the HRRR archive (see documentation below)
        self.load_weather(dateTime = time, filename = out, download_only = False, 
                            nProc = self._Nproc, bounds = bounds)

    def load_weather(self, dateTime = None, filename = None, outDir = None, 
                        download_only = False, nProc = 16, verbose = False, 
                        bounds = None):
        '''
        Consistent class method to be implemented across all weather model types. 
        As a result of calling this method, all of the variables (x, y, z, p, q, 
        t, wet_refractivity, hydrostatic refractivity, e) should be fully 
        populated. 
        '''
        self._load_pressure_levels(dateTime, filename, nProc = nProc, outDir = outDir, 
                                    verbose = verbose)
#        if download_only:
#            self._load_pressure_levels(dateTime, filename, nProc = nProc, outDir = outDir, verbose = verbose)
#        elif filename is not None:
#            try:
#                self._load_pressure_levels(filename = f)
#            except: 
#                self._download_pressure_levels(dateTime, filename=f, nProc = nProc)
#                self._load_pressure_levels(filename = f)
#        else:
#            self._load_pressure_levels(dateTime, nProc = nProc)

        if bounds is not None:
           lat_min, lat_max, lon_min, lon_max = bounds
           self._restrict_model(lat_min, lat_max, lon_min, lon_max)

        self._find_e()
        self._get_wet_refractivity()
        self._get_hydro_refractivity() 
        
        # adjust the grid based on the height data
        self._adjust_grid()



    def _write2HDF5(self, filename, dateTime, verbose = False):
        '''
        Save data from the HRRR archive: doi: 10.7278/S5JQ0Z5B,
        citation: 
             Blaylock B., J. Horel and S. Liston, 2017: Cloud Archiving and 
              Data Mining of High Resolution Rapid Refresh Model Output. 
              Computers and Geosciences. 109, 43-50. 
              doi: 10.1016/j.cageo.2017.08.005.
        '''
        import h5py

        if verbose:
           print('Writing weather model data to HDF5 file {}'.format(filename))

        (Nx, Ny, Npl) = self._ys.shape

        # save the data to an HDF5 file
        today = datetime.datetime.today().date()
        with h5py.File(filename, 'w') as f:
            f.attrs['help'] = 'Raw weather model data from HRRR for {}, accessed {}'.format(dateTime,today)

            pld = f.create_dataset('Pressure_levels', (Npl, ), 'f')
            pld.attrs['help'] = 'Pressure levels'
            pld[:] = np.array(self._pl)

            p = f.create_dataset('Pressure', (Ny, Nx, Npl), 'f')
            p.attrs['help'] = 'pressure grid'
            p[:] = self._p

            lat = f.create_dataset('lats', (Ny, Nx, Npl), 'f')
            lat.attrs['help'] = 'Latitude'
            lat[:] = self._ys 

            lon = f.create_dataset('lons', (Ny, Nx, Npl), 'f')
            lon.attrs['help'] = 'Longitude'
            lon[:] = self._xs

            temp = f.create_dataset('Temperature', (Ny, Nx, Npl), 'f')
            temp.attrs['help'] = 'Temperature'
            temp[:] = self._t

            q = f.create_dataset('Specific_humidity', (Ny, Nx, Npl), 'f')
            q.attrs['help'] = 'Specific humidity'
            q[:] = self._q

            zs = f.create_dataset('Geopotential_height', (Ny, Nx, Npl), 'f')
            zs.attrs['help'] = 'geopotential heights'
            zs[:] = self._zs


        if verbose:
           print('Finished writing weather model data to file')

    def _load_pressure_levels(self, dateTime = None, filename = None, 
                              nProc = 16, outDir = None, verbose = False):
        '''
        Directly load the data from the HRRR archive: doi: 10.7278/S5JQ0Z5B,
        citation: 
             Blaylock B., J. Horel and S. Liston, 2017: Cloud Archiving and 
              Data Mining of High Resolution Rapid Refresh Model Output. 
              Computers and Geosciences. 109, 43-50. 
              doi: 10.1016/j.cageo.2017.08.005.
        '''
        if outDir is None:
            outDir = os.getcwd()
        if filename is  None:
            filename = os.path.join(outDir, 'hrrr_{}.hdf5'.format(dateTime.strftime('%Y%m%d_%H%M%S')))

        if os.path.exists(filename):
           self._ys, self._xs, self._t, self._q, self._zs, self._p = \
                              _load_pressure_levels_from_file(filename)
           return

        lats, lons, temps, qs, geo_hgt, pl = makeDataCubes(dateTime, nProc = nProc, 
                                                  outDir = outDir, verbose = verbose)
        Ny, Nx = lats.shape

        lons[lons > 180] -= 360

        # data cube format should be lons, lats, heights
        _lons = np.broadcast_to(lons[..., np.newaxis],
                                     geo_hgt.shape)
        _lats = np.broadcast_to(lats[..., np.newaxis],
                                     geo_hgt.shape)

        # correct for latitude
        self._get_heights(_lats, geo_hgt)

        self._t = temps
        self._q = qs

        self._p = np.broadcast_to(pl[np.newaxis, np.newaxis, :],
                                  self._zs.shape)
        self._xs = _lons
        self._ys = _lats

        # flip stuff around to match convention
        self._p = np.flip(self._p, axis = 2)
        self._t = np.flip(self._t, axis = 2)
        self._zs = np.flip(self._zs, axis = 2)
        self._q = np.flip(self._q, axis = 2)

        self._write2HDF5(filename)

    def _load_pressure_levels_from_file(self, fileName):
        '''
        Load pre-downloaded weather model from an HDF5 file
        '''
        import h5py

        # load the data
        with h5py.File(fileName, 'r') as f:
            p = f['Pressure'].value.copy()
            z = f['Geopotential_height'].value.copy()
            lats = f['lats'].value.copy()
            lons = f['lons'].value.copy()
            q = f['Specific humidity'].value.copy()
            t = f['Temperature'].value.copy()

        return lats, lons, t, q, z, p
        

def makeDataCubes(dateTime = None, outDir = None, nProc = 16, verbose = False):
    '''
    Create a cube of data representing temperature and relative humidity 
    at specified pressure levels    
    '''
    pl = getPresLevels()
    pl = [convertmb2Pa(p) for p in pl['Values']]

    outName = download_hrrr_file(dateTime, 'hrrr', field = 'prs', outDir = outDir, verbose = verbose)
#    tempList = getTemps(dateTime, pl['Values'], nProc= nProc)
#    rhList = getRH(dateTime, pl['Values'], nProc= nProc)
#    zList = getZ(dateTime, pl['Values'], nProc= nProc)

#    try:
#       temps = stackList(tempList)
#    except:
#       raise RuntimeError('makeDataCubes: Something likely went wrong with the file download')
#
#    rhs = stackList(rhList)
#    zs = stackList(zList)

#    lats, lons = getLatLonsFromList(zList)

    t, z, q, lats, lons = pull_hrrr_data(outName, verbose = False)

    return lats.T, lons.T, t.moveaxes([2, 1, 0]), q.moveaxes([2, 1, 0]), z.moveaxes([2, 1, 0]), pl


def convertmb2Pa(pres):
    return 100*pres

def getLatLonsFromList(List):
    return List[0]['lat'], List[0]['lon']

def stackList(List):
    '''
    Take an input list of variable dictionaries and stack the data into a cube
    '''
    dataList = []
    for d in List:
        dataList.append(d['value'])
    return np.stack(dataList, axis = 2)
        

def getPresLevels():
    presList = [int(v) for v in range(50, 1025, 25)]
    presList.append(1013.2)
    outDict = {'Values': presList, 'units': 'mb', 'Name': 'Pressure_levels'}
    return outDict


def getTemps(dateTime = None, presLevels = None, nProc = 16):
    dateTime = checkDateTime(dateTime)
    varNames = ['TMP:{} mb'.format(val) for val in presLevels]
    List = parfetch(dateTime, varNames, numProc = nProc)
    return List


def getRH(dateTime = None, presLevels = None, nProc = 16):
    dateTime = checkDateTime(dateTime)
    varNames = ['RH:{} mb'.format(val) for val in presLevels]
    List = parfetch(dateTime, varNames, numProc = nProc)
    return List


def getZ(dateTime = None, presLevels = None, nProc = 16):
    dateTime = checkDateTime(dateTime)
    varNames = ['HGT:{} mb'.format(val) for val in presLevels]
    List = parfetch(dateTime, varNames, numProc = nProc)
    return List 


def worker(tup):
    '''
    Helper function for the parallel processing function parfetch
    '''
    dateTime, var, fieldName = tup
    res = get_hrrr_variable(dateTime, var, fxx=0, field=fieldName, model='hrrr')
    return res


def parfetch(dateTime, varNames, fieldName = 'prs', numProc = 1):
    import multiprocessing as mp

    tupList = [(dateTime, var, fieldName) for var in varNames]
    if numProc > 1:
       pool = mp.Pool()
       individual_results = pool.map(worker, tupList)
       pool.close()
       pool.join()
    else:
       individual_results = []
       for tup in tupList:
           print(('Currently fetching variable {}'.format(tup[1])))
           individual_results.append(worker(tup))

    return individual_results


def checkDateTime(dateTime):
    if dateTime is None:
        dateTime = datetime.datetime(2016, 12, 5, 6)
    return dateTime


def pull_hrrr_data(filename, verbose = False):
    '''
    Get the variables from a HRRR grib2 file
    '''
    from cfgrib.xarray_store import open_dataset 

    # open the dataset and pull the data
    ds = open_dataset(filename, backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
    t = ds['t'].values.copy()
    z = ds['gh'].values.copy()
    q = ds['q'].values.copy()

    lats = ds['t'].latitude.values.copy()
    lons = ds['t'].longitude.values.copy()

    del ds

    return t, z, q, lats, lons


def download_hrrr_file(DATE, model, field = 'prs', outDir = None, verbose = False):
    ''' 
    Download a HRRR model
    ''' 
    import requests

    fxx = '00'
    if outDir is None:
       outDir = os.getcwd()
    outfile = '{}_{}_{}_f00.grib2'.format(model, DATE.strftime('%Y%m%d_%H%M%S'), field)
    writeLoc = os.path.join(outDir, outfile)

    grib2file = 'https://pando-rgw01.chpc.utah.edu/{}/{}/{}/{}.t{:02d}z.wrf{}f{:02d}.grib2' \
                    .format(model, field,  DATE.strftime('%Y%m%d'), model, DATE.hour, field, fxx)

    if verbose:
       print('Downloading {} to {}'.format(grib2file, writeLoc))

    r = requests.get(grib2file)
    with open(writeLoc, 'wb') as f:
       f.write(r.content)

    if verbose:
       print('Success!')

    return writeLoc


def interp2D(tup):
    from scipy.interpolate import griddata

    xs, ys, values, xnew, ynew = tup
    new = griddata((xs, ys),values, (xnew, ynew), method = 'linear')
    return new


def interp2DLayers(xs, ys, values, xnew, ynew):
    '''
    Implement a 2D interpolator to transform the non-uniform
    HRRR grid to a uniform lat-long grid. This should be updated
    in future to be avoided. 
    '''
    print('Interpolating to a uniform grid in 2D')
    import multiprocessing as mp

    # set up the parallel interpolation
    tupList = []
    for layerNum, layer in enumerate(np.moveaxis(values, 2, 0)):
        tup = (xs[..., layerNum].flatten(), ys[..., layerNum].flatten(),
               layer.flatten(),xnew, ynew)
        tupList.append(tup)

    pool = mp.Pool(12)
    newLayers = pool.map(interp2D, tupList)
 #   newLayers = []
 #   for tup in tupList:
 #       newLayers.append(interp2D(tup))
    newVals = np.stack(newLayers, axis =2)

    return newVals


def getNewXY(xbounds, ybounds, xstep, ystep):
    '''
    Get new uniform X,Y values from the bounds
    '''
    xnew = np.arange(xbounds[0], xbounds[1], xstep)
    ynew =  np.arange(ybounds[0], ybounds[1], ystep)
    [X, Y] = np.meshgrid(xnew, ynew)
    
    return X, Y

