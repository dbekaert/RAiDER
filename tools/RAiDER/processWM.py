#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import sys

def getWMFilename(weather_model_name, time, outLoc, verbose = False):
    '''
    Check whether the output weather model exists, and
    if not, download it.
    '''
    from datetime import datetime as dt
    from RAiDER.utilFcns import mkdir

    mkdir('weather_files')
    f = os.path.join(outLoc, 
        '{}_{}.nc'.format(weather_model_name,
         dt.strftime(time, '%Y_%m_%d_T%H_%M_%S')))

    if verbose:
        print('Storing weather model at: {}'.format(f))

    download_flag = True
    if os.path.exists(f):
       print('WARNING: Weather model already exists, skipping download')
       download_flag = False

    return download_flag, f


def prepareWeatherModel(weatherDict, wmFileLoc, out, lats=None, lons=None, time=None, verbose = False, download_only = False, makePlots = False):
    '''
    Parse inputs to download and prepare a weather model grid for interpolation
    '''
    import numpy as np
    from RAiDER.models.allowed import checkIfImplemented
    from RAiDER.utilFcns import pickle_load, writeLL, getTimeFromFile
    
    # Make weather
    weather_model, weather_files, weather_model_name = \
               weatherDict['type'],weatherDict['files'],weatherDict['name']
    checkIfImplemented(weather_model_name.upper().replace('-',''))

    # check whether weather model files are supplied
    if weather_files is None:
        download_flag, f = getWMFilename(weather_model.Model(), time, wmFileLoc, verbose)
    else:
        download_flag = False
        time = getTimeFromFile(weather_files[0])

    # if no weather model files supplied, check the standard location
    if download_flag:
        try:
            weather_model.fetch(lats, lons, time, f)
        except Exception as e:
            print('ERROR: Unable to download weather data')
            print('Exception encountered: {}'.format(e))
            sys.exit(0)
 
        # exit on download if download_only requested
        if download_only:
            print('WARNING: download_only flag selected. I will only '
                  'download the weather model, '
                  ' without doing any further processing.')
            return None, None, None

    # Load the weather model data
    if weather_model_name == 'pickle':
        weather_model = pickle_load(weather_files)
    elif weather_files is not None:
        weather_model.load(*weather_files, outLats = lats, outLons = lons)
        download_flag = False
    else:
        weather_model.load(f, outLats = lats, outLons = lons)

    # Pull the lat/lon data if using the weather model 
    if lats is None or len(lats)==2:
        uwn = True
        lats,lons = weather_model.getLL() 
        lla = weather_model.getProjection()
        try:
            writeLL(time, lats, lons,lla, weather_model_name, out)
        except RuntimeError:
            try:
                os.mkdir(os.path.split(weather_model_name)[0])
                writeLL(time, lats, lons,lla, weather_model_name, out)
            except:
                print('Cannot save weather model Lat/Lons')
                print('Continuing to process')
    else:
        uwn = False

    # weather model name
    if verbose:
        print('Number of weather model nodes: {}'.format(np.prod(weather_model.getWetRefractivity().shape)))
        print('Shape of weather model: {}'.format(weather_model.getWetRefractivity().shape))
        print('Bounds of the weather model: {}/{}/{}/{} (SNWE)'
               .format(np.nanmin(weather_model._ys), np.nanmax(weather_model._ys), 
                      np.nanmin(weather_model._xs), np.nanmax(weather_model._xs)))
        print('Using weather nodes only? (true/false): {}'.format(uwn))
        print('Weather model: {}'.format(weather_model.Model()))
        print('Mean value of the wet refractivity: {}'
               .format(np.nanmean(weather_model.getWetRefractivity())))
        print('Mean value of the hydrostatic refractivity: {}'
              .format(np.nanmean(weather_model.getHydroRefractivity())))
        # If the verbose option is called, write out the weather model to a pickle file
        print('Saving weather model object to pickle file')
        import pickle
        pickleFilename = os.path.join(out, 'pickledWeatherModel.pik')
        with open(pickleFilename, 'wb') as f:
           pickle.dump(weather_model, f)
        print('Weather Model Name: {}'.format(weather_model.Model()))
        print(weather_model)

    if makePlots:
        p = weather_model.plot('wh', True)
        p = weather_model.plot('pqt', True)

    return weather_model, lats, lons


