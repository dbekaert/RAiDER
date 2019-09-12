#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
from datetime import datetime as dt

# local imports
from RAiDER.util import mkdir


def getWMFilename(weather_model_name, time, outLoc, verbose = False):
    '''
    Check whether the output weather model exists, and
    if not, download it.
    '''
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


def prepareWeatherModel(lats, lons, time, weatherDict, wmFileLoc, verbose = False, download_only = False):
    '''
    Parse inputs to download and prepare a weather model grid for interpolation
    '''
    from RAiDER.models.allowed import checkIfImplemented
    
    # Make weather
    weather_model, weather_files, weather_model_name = \
               weatherDict['type'],weatherDict['files'],weatherDict['name']
    checkIfImplemented(weather_model_name.upper().replace('-',''))

    # check whether weather model files are supplied
    if weather_files is None:
        download_flag, f = getWMFilename(weather_model.Model(), time, wmFileLoc, verbose)
    else:
        download_flag = False

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
            return None, None

    # Load the weather model data
    if weather_model_name == 'pickle':
        weather_model = RAiDER.util.pickle_load(weather_files)
    elif weather_files is not None:
        weather_model.load(*weather_files)
        download_flag = False
    else:
        # output file for storing the weather model
        #weather_model.load(f)
        weather_model.load(f, lats = lats, lons = lons)

    # weather model name
    if verbose:
        print('Number of weather model nodes: {}'.format(np.prod(weather_model.getWetRefractivity().shape)))
        print('Shape of weather model: {}'.format(weather_model.getWetRefractivity().shape))
        print('Bounds of the weather model: {}/{}/{}/{} (SNWE)'
               .format(np.nanmin(weather_model._ys), np.nanmax(weather_model._ys), 
                      np.nanmin(weather_model._xs), np.nanmax(weather_model._xs)))
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
        p = weather.plot(p, 'rh')
        p.savefig(os.path.join(out, 'rh_plot.pdf'))
        p = weather.plot(p, 'pqt')
        p.savefig(os.path.join(out, 'pqt_plot.pdf'))

    # Pull the lat/lon data if using the weather model 
    if lats is None or len(lats)==2:
        uwn = True
        lats,lons = weather_model.getLL() 
        lla = weather_model.getProjection()
        RAiDER.util.writeLL(time, lats, lons,lla, weather_model_name, out)

    return weather_model, lats, lons

