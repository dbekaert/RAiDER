

import datetime 
import numpy as np
import os
import pyproj
import re
import sys

import util
from weatherModel import WeatherModel

class HRRR(WeatherModel):
    # I took this from
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels.
    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        self._humidityType = 'rh'
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
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)

        # execute the search at the HRRR archive (see documentation below)
        self.load_weather(time, out, download_only = False, nProc = self._Nproc)
        self._restrict_model(lat_min, lat_max, lon_min, lon_max)


    def load_weather(self, dateTime = None, filename = None, download_only = False, nProc = 16):
        '''
        Consistent class method to be implemented across all weather model types. 
        As a result of calling this method, all of the variables (x, y, z, p, q, 
        t, wet_refractivity, hydrostatic refractivity, e) should be fully 
        populated. 
        '''
        if download_only:
            self._download_pressure_levels(dateTime, filename, nProc = nProc)
        elif filename is not None:
            self._load_pressure_levels(filename = filename)
        else:
            self._load_pressure_levels(dateTime, nProc = nProc)


    def _restrict_model(self, lat_min, lat_max, lon_min, lon_max):
        '''
        Restrict the HRRR model to the region of interest (ROI). If there are no 
        points left in the model (because e.g. the ROI is outside the US), raise
        an exception. 
        '''
        mask = (self._xs > lon_min) & (self._xs < lon_max) & (self._ys > lat_min) & (self._ys < lat_max)

        # if there are on points left, raise exception
        if np.sum(mask) == 0:
            raise RuntimeError('Region of interest is outside the region of the HRRR weather archive data')

        # otherwise subset the data
        self._xs = self._xs[mask]
        self._ys = self._ys[mask]
        self._zs = self._zs[mask]
        self._rh = self._rh[mask]
        self._t  = self._t[mask]
        self._p  = self._p[mask]


    def _download_pressure_levels(self, dateTime, filename, nProc = 16):
        '''
        Save data from the HRRR archive: doi: 10.7278/S5JQ0Z5B,
        citation: 
             Blaylock B., J. Horel and S. Liston, 2017: Cloud Archiving and 
              Data Mining of High Resolution Rapid Refresh Model Output. 
              Computers and Geosciences. 109, 43-50. 
              doi: 10.1016/j.cageo.2017.08.005.
        '''
        import h5py

        if filename is  None:
            filename = 'hrrr_{}.hdf5'.format(dateTime.strftime('%Y%m%d'))

        # download the 'raw' data
        lats, lons, temps, rhs, z, pl = makeDataCubes(dateTime, nProc = nProc)
        Npl = len(pl['Values'])
        Ny, Nx = lats.shape

        # save the data to an HDF5 file
        today = datetime.datetime.today().date()
        with h5py.File(filename, 'w') as f:
            f.attrs['help'] = 'Raw weather model data from HRRR for {}, accessed {}'.format(dateTime,today)

            pld = f.create_dataset('Pressure_levels', (Npl, ), 'f')
            pld.attrs['help'] = 'Pressure levels'
            pld[:] = np.array(pl['Values'])

            lat = f.create_dataset('lats', (Ny, Nx), 'f')
            lat.attrs['help'] = 'Latitude'
            lat[:] = lats

            lon = f.create_dataset('lons', (Ny, Nx), 'f')
            lon.attrs['help'] = 'Longitude'
            lon[:] = lons

            temp = f.create_dataset('Temperature', (Ny, Nx, Npl), 'f')
            temp.attrs['help'] = 'Temperature'
            temp[:] = temps

            rh = f.create_dataset('Relative_humidity', (Ny, Nx, Npl), 'f')
            rh.attrs['help'] = 'Relative humidity %'
            rh[:] = rhs

            zs = f.create_dataset('Geopotential_height', (Ny, Nx, Npl), 'f')
            zs.attrs['help'] = 'geopotential heights'
            zs[:] = z 


    def _load_pressure_levels(self, dateTime = None, filename = None, nProc = 16):
        '''
        Directly load the data from the HRRR archive: doi: 10.7278/S5JQ0Z5B,
        citation: 
             Blaylock B., J. Horel and S. Liston, 2017: Cloud Archiving and 
              Data Mining of High Resolution Rapid Refresh Model Output. 
              Computers and Geosciences. 109, 43-50. 
              doi: 10.1016/j.cageo.2017.08.005.
        '''
        if filename is not None:
            lats, lons, temps, rhs, z, pl = self._load_pressure_levels_from_file(filename)
        else:
            lats, lons, temps, rhs, z, pl = makeDataCubes(dateTime, nProc = nProc)

        lons = lons.T
        lats = lats.T
        rhs  = np.swapaxes(rhs, 0, 1)
        z    = np.swapaxes(z, 0, 1)
        temps= np.swapaxes(temps, 0, 1)

        lons[lons > 180] -= 360
        self._proj = pyproj.Proj(proj='latlong')

        # data cube format should be lons, lats, heights
        geo_hgt = z/self._g0
        _lons = np.broadcast_to(lons[..., np.newaxis],
                                     geo_hgt.shape)
        _lats = np.broadcast_to(lats[..., np.newaxis],
                                     geo_hgt.shape)

        # correct for latitude
        self._get_heights(_lats, geo_hgt)

        self._t = temps
        self._rh = rhs

        self._p = np.broadcast_to(pl[np.newaxis, np.newaxis, :],
                                  self._zs.shape)

        self._xs = _lons
        self._ys = _lats


    def _load_pressure_levels_from_file(self, fileName):
        '''
        Load pre-downloaded weather model from an HDF5 file
        '''
        import h5py

        # load the data
        with h5py.File(fileName, 'r') as f:
            pl = f['Pressure_levels'].value.copy()
            z = f['Geopotential_height'].value.copy()
            lats = f['lats'].value.copy()
            lons = f['lons'].value.copy()
            rh = f['Relative_humidity'].value.copy()
            t = f['Temperature'].value.copy()

        return lats, lons, t, rh, z, pl
        

def makeDataCubes(dateTime = None, nProc = 16):
    '''
    Create a cube of data representing temperature and relative humidity 
    at specified pressure levels    
    '''
    pl = getPresLevels()

    tempList = getTemps(dateTime, pl['Values'], nProc= nProc)
    rhList = getRH(dateTime, pl['Values'], nProc= nProc)
    zList = getZ(dateTime, pl['Values'], nProc= nProc)

    try:
       temps = stackList(tempList)
    except:
       raise RuntimeError('makeDataCubes: Something likely went wrong with the file download')

    rhs = stackList(rhList)
    zs = stackList(zList)

    lats, lons = getLatLonsFromList(zList)

    return lats, lons, temps, rhs, zs, pl


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


def parfetch(dateTime, varNames, fieldName = 'prs', numProc = 16):
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


# the following function is modified from the repository at 
# https://github.com/blaylockbk/pyBKB_v2/tree/master/BB_downloads
def get_hrrr_variable(DATE, variable,
                      fxx=0,
                      model='hrrr',
                      field='sfc',
                      removeFile=True,
                      value_only=False,
                      verbose=False,
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


