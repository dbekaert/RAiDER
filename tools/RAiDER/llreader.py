#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import os

from RAiDER.demdownload import download_dem
from RAiDER.utilFcns import gdal_open

def readLL(*args):
    '''
    Parse lat/lon/height inputs and return
    the appropriate outputs
    '''
    if len(args)==2:
       flag='files'
    elif len(args)==4:
       flag='bounding_box'
    elif len(args)==1:
       flag = 'station_list'
    else:
       raise RuntimeError('llreader: Cannot parse args')

    # Lats/Lons
    if flag=='files':
        # If they are files, open them
        lat, lon = args
        lats, latproj, lat_gt = gdal_open(lat, returnProj = True)
        lons, lonproj, lon_gt = gdal_open(lon, returnProj = True)
    elif flag=='bounding_box':
        N,W,S,E = args
        lats = np.array([float(N), float(S)])
        lons = np.array([float(E), float(W)])
        latproj = lonproj = 'EPSG:4326'
        if (lats[0] == lats[1]) | (lons[0]==lons[1]):
           raise RuntimeError('You have passed a zero-size bounding box: {}'
                               .format(args.bounding_box))
    elif flag=='station_list':
        lats, lons = readLLFromStationFile(*args)
        latproj = lonproj = 'EPSG:4326'
    else:
        # They'll get set later with weather
        lats = lons = None
        latproj = lonproj = None
        #raise RuntimeError('readLL: unknown flag')

    [lats, lons] = enforceNumpyArray(lats, lons)

    return lats, lons, latproj, lonproj


def getHeights(lats, lons,heights, demFlag = 'dem'):
    '''
    Fcn to return heights from a DEM, either one that already exists
    or will download one if needed.
    '''
    height_type, demFilename = heights

    if height_type == 'dem':
      try:
          hts = gdal_open(demFilename)
      except:
          print('WARNING: File {} could not be opened. \n'.format(demFilename))
          print('Proceeding with DEM download')
          height_type = 'download'

    elif height_type == 'lvs':
        hts = demFilename
        latlist, lonlist, hgtlist = [], [], []
        for ht in hts:
            latlist.append(lats.flatten())
            lonlist.append(lons.flatten())
            hgtlist.append(np.array([ht]*len(lats.flatten())))
        lats = np.array(latlist)
        lons = np.array(lonlist)
        hts = np.array(hgtlist)

    elif height_type == 'merge':
        import pandas as pd
        for f in demFilename:
            data = pd.read_csv(f)
            lats = data['Lat'].values
            lons = data['Lon'].values
            hts = download_dem(lats, lons, outName = f, save_flag = 'merge')
    else:
        height_type = 'download'
        
    if height_type == 'download':
        hts = download_dem(lats, lons, outName = os.path.abspath(demFilename))

    [lats, lons, hts] = enforceNumpyArray(lats, lons, hts)

    return lats, lons, hts


def setGeoInfo(lat, lon, latproj, lonproj, outformat):
    # TODO: implement
    # set_geo_info should be a list of functions to call on the dataset,
    # and each will do some bit of work
    set_geo_info = list()
    if lat is not None:
        def geo_info(ds):
            ds.SetMetadata({'X_DATASET': os.path.abspath(lat), 'X_BAND': '1',
                            'Y_DATASET': os.path.abspath(lon), 'Y_BAND': '1'})
        set_geo_info.append(geo_info)
    # Is it ever possible that lats and lons will actually have embedded
    # projections?
    if latproj:
        if outformat != 'h5':
            def geo_info(ds):
                ds.SetProjection(latproj)
        else:
            geo_info = None
    elif lonproj:
        def geo_info(ds):
            ds.SetProjection(lonproj)
        set_geo_info.append(geo_info)

    return set_geo_info


def enforceNumpyArray(*args):
    '''
    Enforce that a set of arguments are all numpy arrays. 
    Raise an error on failure.
    '''
    return [checkArg(a) for a in args]

def checkArg(arg):

    if arg is None:
       return None
    else:
       import numpy as np
       try:
          return np.array(arg)
       except:
          raise RuntimeError('checkArg: Cannot covert argument to numpy arrays')


def readLLFromStationFile(fname):
    '''
    Helper fcn for checking argument compatibility
    '''
    try:
       import pandas as pd
       stats = pd.read_csv(fname)
       return stats['Lat'].values,stats['Lon'].values
    except:
       lats, lons = [], []
       with open(fname, 'r') as f:
          for i, line in enumerate(f): 
              if i == 0:
                 continue
              lat, lon = [float(f) for f in line.split(',')[1:3]]
              lats.append(lat)
              lons.append(lon)
       return lats, lons
