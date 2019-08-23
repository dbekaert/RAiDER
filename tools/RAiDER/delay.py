#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently we take samples every _STEP meters, which causes either
inaccuracies or inefficiencies, and we no longer can integrate to
infinity.
"""

# standard imports
import numpy as np
import os
import sys

# local imports
from RAiDER.constants import Zenith, _ZMAX, _STEP
import RAiDER.losreader as losreader
import RAiDER.util as util
from RAiDER.downloadWM import downloadWMFile as dwf
import RAiDER.delayFcns as df


def _common_delay(weatherObj, lats, lons, heights, 
                  look_vecs, zref = None, useWeatherNodes = False,
                  stepSize = _STEP, interpType = 'rgi',
                  verbose = False, nproc = 8, useDask = False):
    """
    This function calculates the line-of-sight vectors, estimates the point-wise refractivity
    index for each one, and then integrates to get the total delay in meters. The point-wise
    delay is calculated by interpolating the weatherObj, which contains a weather model with
    wet and hydrostatic refractivity at each weather model grid node, to the points along 
    the ray. The refractivity is integrated along the ray to get the final delay. 

    Inputs: 
     weatherObj - a weather model object
     lats       - Grid of latitudes for each ground point
     lons       - Grid of longitudes for each ground point
     heights    - Grid of heights for each ground point
     look_vecs  - Grid of look vectors (should be full-length) for each ground point
     raytrace   - If True, will use the raytracing method, if False, will use the Zenith 
                  + projection method
     stepSize   - Integration step size in meters 
     intpType   - Can be one of 'scipy': LinearNDInterpolator, or 'sane': _sane_interpolate. 
                  Any other string will use the RegularGridInterpolate method
     nproc      - Number of parallel processes to use if useDask is True
     useDask    - use Dask to parallelize ray calculation

    Outputs: 
     delays     - A list containing the wet and hydrostatic delays for each ground point in 
                  meters. 
    """
    if verbose:
       import time as timing

    parThresh = 1e5

    # If the number of points to interpolate are low, don't parallelize
    if np.prod(lats.shape) < parThresh:
       useDask = False
       nproc = 1

    # Determine if/what type of parallization to use
    if useDask:
       import dask.bag as db
    elif nproc > 1:
       import multiprocessing as mp
    else:
       pass

    # If weather model nodes only are desired, the calculation is very quick
    if look_vecs is Zenith and useWeatherNodes:
        _,_,zs = weatherObj.getPoints()
        look_vecs = _getZenithLookVecs(lats, lons, heights, zref = zref)
        wet_pw  = weatherObj.getWetRefractivity()
        hydro_pw= weatherObj.getHydroRefractivity()
        wet_delays = _integrateZenith(zs, wet_pw)
        hydro_delays = _integrateZenith(zs, hydro_pw)
        return wet_delays,hydro_delays

    rays, ecef = df.calculate_rays(lats, lons, heights, look_vecs, 
                     zref = zref, stepSize = stepSize, verbose=verbose)

    newProj = weatherObj.getProjection()
    newPts = [df._transform(ray, ecef, newProj) for ray in rays]
    rays = []
    for pnt in newPts:
       rays.append(np.array([pnt[:,1], pnt[:,0], pnt[:,2]]).T)

    # Define the interpolator objects
    ifWet = getIntFcn(weatherObj,interpType =interpType)
    ifHydro = getIntFcn(weatherObj,itype = 'hydro', interpType = interpType)

    if verbose:
        print('Beginning interpolation of each ray')
        st = timing.time()

    if useDask:
        Npart = min(len(newPts)//100 + 1, 1000)
        PntBag = db.from_sequence(newPts, npartitions=Npart)
        wet_pw = PntBag.map(interpRay).compute()
        hydro_pw = PntBag.map(interpRay).compute()
    elif nproc > 1:
        import multiprocessing as mp
        pool = mp.Pool(12)
        inp1 = zip([ifWet]*len(newPts), newPts)
        inp2 = zip([ifHydro]*len(newPts), newPts)

        wet_pw = pool.map(interpRay,inp1)
        hydro_pw = pool.map(interpRay, inp2)
    else:
        wet_pw, hydro_pw = [], []
        count = 0
        for ray in rays:
            wet_pw.append(interpRay((ifWet, ray)))
            hydro_pw.append(interpRay((ifHydro, ray)))
            count = count+1
       
  
    if verbose:
        print('_common_delay: Finished interpolation')
        ft = timing.time()
        print('Interpolation took {:4.2f} secs'.format(ft-st))
        print('Average of {:1.6f} secs/ray'.format(.5*(ft-st)/len(newPts)))
        print('_common_delay: finished point-wise delay calculations')

    # intergrate the point-wise delays to get total slant delay
    delays = _integrateLOS(stepSize, wet_pw, hydro_pw)

    return delays


def getIntFcn(weatherObj, itype = 'wet', interpType = 'scipy'):
    '''
    Function to create and return an Interpolator object
    '''
    import RAiDER.interpolator as intprn

    ifFun = intprn.Interpolator()
    ifFun.setPoints(*weatherObj.getPoints())
    ifFun.setProjection(weatherObj.getProjection())
    import pdb; pdb.set_trace()

    if itype == 'wet':
        ifFun.getInterpFcns(weatherObj.getWetRefractivity().filled(fill_value=np.nan), interpType = interpType)
    elif itype == 'hydro':
        ifFun.getInterpFcns(weatherObj.getHydroRefractivity().filled(fill_value=np.nan), interpType = interpType)
    return ifFun


# call the interpolator on each ray
def interpRay(tup):
    fcn, ray = tup
    return fcn(ray)[0]


def _integrateLOS(stepSize, wet_pw, hydro_pw):
    delays = [] 
    for d in (wet_pw, hydro_pw):
        delays.append(_integrate_delays(stepSize, d))
    return delays


def _integrateZenith(zs, pw):
    return 1e-6*np.trapz(pw, zs, axis = 2)


def _integrate_delays(stepSize, refr):
    '''
    This function gets the actual delays by integrating the refractivity in 
    each node. Refractivity is given in the 'refr' variable. 
    '''
    delays = []
    for ray in refr:
        delays.append(int_fcn(ray, stepSize))
    return delays


# integrate the delays to get overall delay
def int_fcn(y, dx):
    return 1e-6*dx*np.nansum(y)


def tropo_delay(time, los = None, lats = None, lons = None, heights = None, 
                weather = None, wmFileLoc = None, zref = 15000, out = None, 
                parallel=True,verbose = False, download_only = False):
    """Calculate troposphere delay from command-line arguments.

    We do a little bit of preprocessing, then call
    _common_delay. 
    """
    from RAiDER.models.allowed import checkIfImplemented
    from datetime import datetime as dt
    from RAiDER.llreader import getHeights

    if verbose:
        print('type of time: {}'.format(type(time)))
        print('Download-only is {}'.format(download_only))

    # location of the weather model files
    if wmFileLoc is None:
       wmFileLoc = os.path.join(out, 'weather_files')

    # ensuring consistent file extensions
    #outformat = output_format(outformat)

    # the output folder where data is downloaded and delays are stored, default is same location
    if out is None:
        out = os.getcwd()

    # Make weather
    weather_model, weather_files, weather_model_name = \
               weather['type'],weather['files'],weather['name']
    checkIfImplemented(weather_model_name)

    # check whether weather model files are supplied
    if weather_files is None:
       download_flag, f = dwf(weather_model.Model(), time, wmFileLoc, verbose)

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
            print('WARNING: download_only flag selected. I will only '\
                  'download the weather model, '\
                  ' without doing any further processing.')
            return None, None


    # Load the weather model data
    if weather_files is not None:
       weather_model.load(*weather_files)
       download_flag = False
    elif weather_model_name == 'pickle':
        weather_model = util.pickle_load(weather_files)
    else:
        # output file for storing the weather model
        #weather_model.load(f)
        weather_model.load(f, lats = lats, lons = lons)

    # weather model name
    if verbose:
        print('Weather Model Name: {}'.format(weather_model.Model()))
        #p = weather.plot(p)

    # Pull the lat/lon data if using the weather model 
    if lats is None or len(lats)==2:
       lats,lons = weather_model.getLL() 
       lla = weather_model.getProjection()
       util.writeLL(time, lats, lons,lla, weather_model_name, out)
    
    # check for compatilibility of the weather model locations and the input
    if util.isOutside(util.getExtent(lats, lons), util.getExtent(*weather_model.getLL())):
       print('WARNING: some of the requested points are outside of the existing \
             weather model; these will end up as NaNs')
 
    # Pull the DEM
    if verbose: 
       print('Beginning DEM calculation')
    demLoc = os.path.join(out, 'geom')
    lats, lons, hgts = getHeights(lats, lons,heights, demLoc)

    # LOS check and load
    if verbose: 
       print('Beginning LOS calculation')
    if los is None:
        los = Zenith
    elif los is Zenith:
        pass
    else:
        import utils.losreader as losr
        los = losr.infer_los(los, lats, lons, hgts, zref)

    if los is Zenith:
       raytrace = False
    else:
       raytrace = True
       
    util.checkShapes(los, lats, lons, hgts)
    util.checkLOS(los, raytrace, np.prod(lats.shape))

    # Save the shape so we can restore later, but flatten to make it
    # easier to think about
    llas = np.stack((lats, lons, hgts), axis=-1)
    real_shape = llas.shape[:-1]
    llas = llas.reshape(-1, 3)
    lats, lons, hgts = np.moveaxis(llas, -1, 0)

    if verbose: 
       print('Beginning delay calculation')
    # Call _common_delay to compute the hydrostatic and wet delays
    if parallel:
       useDask = True
       nproc = 16
    else:
       useDask = False
       nproc = 1
    wet, hydro = _common_delay(weather_model, lats, lons, hgts, los, zref = zref,\
                  nproc = nproc, useDask = useDask, verbose = verbose)
    if verbose: 
       print('Finished delay calculation')

    # Restore shape
    try:
        hydro, wet = np.stack((hydro, wet)).reshape((2,) + real_shape)
    except:
        pass

    return wet, hydro

    
