

import datetime 
import numpy as np
import os
import pyproj
import re
import sys

import RAiDER.util as util
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


    def load_weather(self, dateTime = None, filename = None, 
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
           self._ys, self._xs, self._t, self._q, self._zs, self._p = _load_pressure_levels_from_file(fileName)
           return

        lats, lons, temps, qs, geo_hgt, pl = makeDataCubes(dateTime, nProc = nProc, 
                                                  outDir = outDir, verbose = verbose)
        Npl = len(pl)
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

        return lats, lons, t, q, z, pl
        

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

    

# the following function is modified from the repository at 
# https://github.com/blaylockbk/pyBKB_v2/tree/master/BB_downloads
def get_hrrr_variable(DATE, variable,
                      fxx=0,
                      model='hrrr',
                      field='sfc',
                      removeFile=False,
                      value_only=False,
                      verbose=True,
                      outDIR='./'):
    """
    Uses cURL to grab the requested variable from a HRRR grib2 file in the
    HRRR archive. Uses the the requested variable string to search the .idx
    file and determine the byte range. When the byte range of a variable is
    known, cURL is capable of downloading a single variable from a larger GRIB2
    file. This function packages the data in a dictionary.
    Input:
        DATE       - The datetime(year, month, day, hour) for the HRRR file you
                     want. This is the same as the model run time, in UTC.
        variable   - A string describing the variable you are looking for in the
                     GRIB2 file. Refer to the .idx files. For example:
                        https://pando-rgw01.chpc.utah.edu/hrrr/sfc/20180101/hrrr.t00z.wrfsfcf00.grib2.idx
                     You want to put the variable short name and the level
                     information. For example, for 2m temperature:

                        variable='TMP:2 m above ground'
        fxx        - The forecast hour you desire. Default is the analysis hour,
                     or f00.
        model      - The model you want. Options include ['hrrr', 'hrrrX', 'hrrrak']
        field      - The file output type. Options include ['sfc', 'prs']
        removeFile - True: remove the GRIB2 file after it is downloaded
                     False: do not remove the GRIB2 file after it is downloaded
        value_only - True: only return the values, not the lat/lon.
                        Returns output in 0.2 seconds
                     False: returns value and lat/lon, grib message, analysis and valid datetime.
                        Returns output in 0.75-1 seconds
        verbose    - Prints some diagnostics
        outDIR     - Specify where the downloaded data should be downloaded.
                     Default is the current directory. 
    Tips:
        1. The DATE you request represents the model run time. If you want to
           retrieve the file based on the model's valid time, you need to
           offset the DATE with the forecast lead time. For example:
                VALID_DATE = datetime(year, month, day, hour)   # We want the model data valid at this time
                fxx = 15                                        # Forecast lead time
                RUN_DATE = VALID_DATE-timedelta(hours=fxx)      # The model run datetime that produced the data
                zet_hrrr_variable(RUN_DATE, 'TMP:2 m', fxx=fxx) # The returned data will be a forecast for the requested valid time and lead time
        
        2. You can request both U and V components at a level by using
                variable='UVGRD:10 m'
            This special request will return the U and V component winds
            converted from grid-relative to earth-relative, as well as the 
            calculated wind speed.
            Note: You can still get the grid-relative winds by requesting both
                  'UGRD:10 m' and 'VGRD:10 m' individually.
    """
    import pygrib
    import urllib.request, urllib.error, urllib.parse
    import ssl

    ## --- Catch Errors -------------------------------------------------------
    # Check that you requested the right model name and field name
    if model not in ['hrrr', 'hrrrX', 'hrrrak']:
        raise ValueError("Requested model must be 'hrrr', 'hrrrX', or 'hrrrak'")
    if field not in ['prs', 'sfc']:
        raise ValueError("Requested field must be 'prs' or 'sfc'. We do not store other fields in the archive")

    # Check that you requested the right forecasts available for the model
    if model == 'hrrr' and fxx not in list(range(19)):
        raise ValueError("HRRR: fxx must be between 0 and 18\nYou requested f%02d" % fxx)
    elif model == 'hrrrX' and fxx != 0:
        raise ValueError("HRRRx: fxx must be 0. We do not store other forecasts in the archive.\nYou requested f%02d" % fxx)
    elif model == 'hrrrak' and fxx not in list(range(37)):
        raise ValueError("HRRRak: fxx must be between 0 and 37\nYou requested f%02d" % fxx)

    # Check that the requested hour exists for the model
    if model == 'hrrrak' and DATE.hour not in list(range(0,24,3)):
        raise ValueError("HRRRak: DATE.hour must be 0, 3, 6, 9, 12, 15, 18, or 21\nYou requested %s" % DATE.hour)

    if verbose:
        # Check that the request datetime has happened
        if DATE > datetime.datetime.utcnow():
            print("Warning: The datetime you requested hasn't happened yet\nDATE: %s F%02d\n UTC: %s" % (DATE, fxx, datetime.datetime.utcnow()))
    ## ---(Catch Errors)-------------------------------------------------------


    ## --- Set Temporary File Name --------------------------------------------
    # Temporary file name has to be unique, or else when we use multiprocessing
    # we might accidentally delete files before we are done with them.
    outfile = '%stemp_%s_%s_f%02d_%s.grib2' % (outDIR, model, DATE.strftime('%Y%m%d%H'), fxx, variable[:3].replace(":", ''))

    if verbose is True:
        print(' >> Dowloading tempfile: %s' % outfile)

    ## --- Requested Variable -------------------------------------------------
    # A special variable request is 'UVGRD:[level]' which will get both the U
    # and V wind components converted to earth-relative direction in a single
    # download. Since UGRD always proceeds VGRD, we will set the get_variable
    # as UGRD. Else, set get_variable as variable.
    if variable.split(':')[0] == 'UVGRD':
        # We need both U and V to convert winds from grid-relative to earth-relative
        get_variable = 'UGRD:' + variable.split(':')[1]
    else:
        get_variable = variable


    ## --- Set Data Source ----------------------------------------------------
    """
    Dear User,
      Only HRRR files are only downloaded and added to Pando every 3 hours.
      That means if you are requesting data for today that hasn't been copied
      to Pando yet, you will need to get it from the NOMADS website instead.
      But good news! It's an easy fix. All we need to do is redirect you to the
      NOMADS server. I'll check that the date you are requesting is not for
      today's date. If it is, then I'll send you to NOMADS. Deal? :)
                                                  -Sincerely, Brian
    """

    # If the datetime requested is less than six hours ago, then the file is 
    # most likely on Pando. Else, download from NOMADS. 
    #if DATE+timedelta(hours=fxx) < datetime.utcnow()-timedelta(hours=6):
    if DATE < datetime.datetime.utcnow()-datetime.timedelta(hours=12):
        # Get HRRR from Pando
        if verbose:
            print("Oh, good, you requested a date that should be on Pando.")
        grib2file = 'https://pando-rgw01.chpc.utah.edu/%s/%s/%s/%s.t%02dz.wrf%sf%02d.grib2' \
                    % (model, field,  DATE.strftime('%Y%m%d'), model, DATE.hour, field, fxx)
        fileidx = grib2file+'.idx'
    else:
        # Get operational HRRR from NOMADS
        if model == 'hrrr':
            if verbose:
                print("/n---------------------------------------------------------------------------")

                print("!! Hey! You are requesting a date that is not on the Pando archive yet.  !!")
                print("!! That's ok, I'll redirect you to the NOMADS server. :)                 !!")
                print("---------------------------------------------------------------------------\n")
            #grib2file = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.%s/%s.t%02dz.wrf%sf%02d.grib2' \
            #            % (DATE.strftime('%Y%m%d'), model, DATE.hour, field, fxx)
            grib2file = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.%s/conus/hrrr.t%02dz.wrf%sf%02d.grib2' \
                        % (DATE.strftime('%Y%m%d'), DATE.hour, field, fxx)
            fileidx = grib2file+'.idx'
        elif model == 'hrrrX':
            print("\n-------------------------------------------------------------------------")
            print("!! Sorry, I haven't download that Experimental HRRR run from ESRL yet  !!")
            print("!! Try again in a few hours.                                           !!")
            print("/n---------------------------------------------------------------------------")
            print("!! Hey! You are requesting a date that is not on the Pando archive yet.  !!")
            print("!! That's ok, I'll redirect you to the PARALLEL NOMADS server. :)        !!")
            print("---------------------------------------------------------------------------\n")
            grib2file = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.%s/alaska/hrrr.t%02dz.wrf%sf%02d.ak.grib2' \
                        % (DATE.strftime('%Y%m%d'), DATE.hour, field, fxx)
            fileidx = grib2file+'.idx'

    if verbose:
        print('GRIB2 File: %s' % grib2file)
        print(' .idx File: %s' % fileidx)
        print("")


    ## --- Download Requested Variable ----------------------------------------
    try:
        ## 0) Read the grib2.idx file
        try:
            # ?? Ignore ssl certificate (else urllib2.openurl wont work).
            #    Depends on your version of python.
            #    See here:
            #    http://stackoverflow.com/questions/19268548/python-ignore-certicate-validation-urllib2
            ctx = ssl.create_default_context()

            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            idxpage = urllib.request.urlopen(fileidx, context=ctx)
        except:
            idxpage = urllib.request.urlopen(fileidx)

        lines = [t.decode('utf-8') for t in idxpage.readlines()]

        ## 1) Find the byte range for the requested variable. First find where
        #     in the .idx file the variable is located. We need the byte number
        #     The variable begins on. Keep a count (gcnt) of the line number so
        #     we can also get the beginning byte of the next variable. This is 
        #     our byte range.
        gcnt = 0
        for g in lines:
            expr = re.compile(get_variable)
            if expr.search(g):
                if verbose is True:
                    print(' >> Matched a variable: ', g)
                parts = g.split(':')
                rangestart = parts[1]
                if variable.split(':')[0] == 'UVGRD':
                    parts = lines[gcnt+2].split(':')      # Grab range between U and V variables
                else:
                    parts = lines[gcnt+1].split(':')      # Grab range for requested variable only
                rangeend = int(parts[1])-1
                if verbose is True:
                    print(' >> Byte Range:', rangestart, rangeend)
                byte_range = str(rangestart) + '-' + str(rangeend)
            gcnt += 1
        ## 2) When the byte range is discovered, use cURL to download the file.
        os.system('curl -s -o %s --range %s %s' % (outfile, byte_range, grib2file))


        ## --- Convert winds to earth-relative --------------------------------
        # If the requested variable is 'UVGRD:[level]', then we have to change
        # the wind direction from grid-relative to earth-relative.
        # You can still get the grid-relative winds by requesting 'UGRD:[level]'
        # and # 'VGRD:[level] independently.
      # and # 'VGRD:[level] independently.
        # !!! See more information on why/how to do this here:
        # https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb
        #from wind import wind_uv_to_spd
        if variable.split(':')[0] == 'UVGRD':
            if verbose:
                print(' >> Converting winds to earth-relative')
            wgrib2 = '/uufs/chpc.utah.edu/sys/installdir/wgrib2/2.0.2/wgrib2/wgrib2'
            if model == 'hrrrak':
                regrid = 'nps:225.000000:60.000000 185.117126:1299:3000.000000 41.612949:919:3000.000000'
            if model == 'hrrr' or model == 'hrrrX':
                regrid = 'lambert:262.500000:38.500000:38.500000:38.500000 237.280472:1799:3000.000000 21.138123:1059:3000.000000'
            os.system('%s %s -new_grid_winds earth -new_grid %s %s.earth' % (wgrib2, outfile, regrid, outfile))
            os.system('rm -f %s' % outfile) # remove the original file
            outfile = outfile+'.earth'      # assign the `outfile`` as the regridded file


        ## 3) Get data from the file, using pygrib and return what we want to use
        grbs = pygrib.open(outfile)

        # Note: Returning only the variable value is a bit faster than returning 
        #       the variable value with the lat/lon and other details. You can
        #       specify this when you call the function.
        if value_only:
            if variable.split(':')[0] == 'UVGRD':
                return_this = {'UGRD': grbs[1].values,
                                'VGRD': grbs[2].values,
                                'SPEED': wind_uv_to_spd(grbs[1].values, grbs[2].values)}
            else:
                return_this = {'value': grbs[1].values}
            if removeFile:
                os.system('rm -f %s' % (outfile))
            return return_this
        else:
            if variable.split(':')[0] == 'UVGRD':
                value1, lat, lon = grbs[1].data()
                if model == 'hrrrak':
                    lon[lon>0] -= 360
                return_this = {'UGRD': value1,
                               'VGRD': grbs[2].values,
                               'SPEED': wind_uv_to_spd(value1, grbs[2].values),
                               'lat': lat,
                               'lon': lon,
                               'valid': grbs[1].validDate,
                               'anlys': grbs[1].analDate,
                               'msg': [str(grbs[1]), str(grbs[2])],
                               'name': [grbs[1].name, grbs[2].name],
                               'units': [grbs[1].units, grbs[2].units],
                               'level': [grbs[1].level, grbs[2].level],
                               'URL': grib2file}
            else:
                value, lat, lon = grbs[1].data()
                if model == 'hrrrak':
                    lon[lon>0] -= 360
                return_this = {'value': value,
                               'lat': lat,
                               'lon': lon,
                               'valid': grbs[1].validDate,
                               'anlys': grbs[1].analDate,
                               'msg': str(grbs[1]),
                               'name': grbs[1].name,
                               'units': grbs[1].units,
                               'level': grbs[1].level,
                               'URL': grib2file}
            if removeFile:
                os.system('rm -f %s' % (outfile))

            return return_this


    except:
        if verbose:
            print(" _______________________________________________________________")
            print(" !!   Run Date Requested :", DATE, "F%02d" % fxx)
            print(" !! Valid Date Requested :", DATE+datetime.timedelta(hours=fxx))
            print(" !! Valid Date Requested :", DATE+datetime.timedelta(hours=fxx))
            print(" !!     Current UTC time :", datetime.datetime.utcnow())
            print(" !! ------------------------------------------------------------")
            print(" !! ERROR downloading GRIB2:", grib2file)
            print(" !! Is the variable right?", variable)
            print(" !! Does the .idx file exist?", fileidx)
            print(" ---------------------------------------------------------------")
        return {'value' : np.nan,
                'lat' : np.nan,
                'lon' : np.nan,
                'valid' : np.nan,
                'anlys' : np.nan,
                'msg' : np.nan,
                'URL': grib2file}


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

    
