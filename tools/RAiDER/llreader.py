#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import os

import RAiDER.demdownload as dld
import RAiDER.util as util

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
        lats, latproj = util.gdal_open(lat, returnProj = True)
        lons, lonproj = util.gdal_open(lon, returnProj = True)
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

    return lats, lons


def getHeights(lats, lons,heights, demLoc = None, demFlag = 'dem'):
    # Height
    if demLoc is None:
       demLoc = os.getcwd()

    height_type, height_info = heights

    if height_type == 'dem':
      try:
        hts = util.gdal_open(os.path.join(demLoc, height_info))
      except RuntimeError:
        try:
          import pandas as pd
          data = pd.read_csv(os.path.join(demLoc, 'warpedDEM.dem'))
          hts = data['DEM_hgt_m'].values
        except:
          print('WARNING: File {} could not be opened. \n')
          print('Proceeding with DEM download'.format(height_info))
          height_type = 'download'
    elif height_type == 'lvs':
      hts = height_info
      latlist, lonlist, hgtlist = [], [], []
      for ht in hts:
         latlist.append(lats.flatten())
         lonlist.append(lons.flatten())
         hgtlist.append(np.array([ht]*length(lats.flatten())))
      lats = np.array(latlist)
      lons = np.array(lonlist)
      hts = np.array(hgtlist)
        
    if height_type == 'download':
        hts = dld.download_dem(lats, lons, demLoc)

    [lats, lons, hts] = enforceNumpyArray(lats, lons, hts)

    return lats, lons, hts


def setGeoInfo(lat, lon, latproj, lonproj):
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
        if outformat is not 'h5':
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

       
