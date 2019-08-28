#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import os
import sys

from RAiDER.constants import Zenith,_STEP,_ZREF


def _common_delay(weatherObj, lats, lons, heights, 
                  look_vecs, zref = _ZREF, useWeatherNodes = False,
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
    import RAiDER.delayFcns

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
        look_vecs = RAiDER.delayFcns._getZenithLookVecs(lats, lons, heights, zref = zref)
        wet_pw  = weatherObj.getWetRefractivity()
        hydro_pw= weatherObj.getHydroRefractivity()
        wet_delays = _integrateZenith(zs, wet_pw)
        hydro_delays = _integrateZenith(zs, hydro_pw)
        return wet_delays,hydro_delays

    if verbose:
        print('Beginning ray calculation')
        print('ZREF = {}'.format(zref))
        print('stepSize = {}'.format(stepSize))
        print('# of rays = {}'.format(len(lats)))

    rays, ecef = RAiDER.delayFcns.calculate_rays(lats, lons, heights, look_vecs, 
                     zref = zref, stepSize = stepSize, verbose=verbose)

    if verbose:
        print('First ten points along first ray: {}'.format(rays[0][:10,:]))
        print('First ten points along last ray: {}'.format(rays[-1][:10,:]))

    wmProj = weatherObj.getProjection()
    newRays = reprojectRays(rays, ecef, wmProj)
    rays = orderForInterp(newRays)

    if verbose:
        print('Finished ray calculation')
        print('First ten points along first ray: {}'.format(rays[0][:10,:]))
        print('First ten points along last ray: {}'.format(rays[-1][:10,:]))
        try:
            print('NaN check: {}'.format(['PASSED' if np.sum(np.isnan(rays[0]))==0 else 'FAILED'][0]))
        except:
            print('Ray 1 has length 0')

    # Define the interpolator objects
    ifWet = getIntFcn(weatherObj,interpType =interpType)
    ifHydro = getIntFcn(weatherObj,itype = 'hydro', interpType = interpType)

    if verbose:
       print('Wet interpolator bounding box: {}'.format(ifWet._bbox))
       print('Hydrostatic interpolator bounding box: {}'.format(ifHydro._bbox))

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
        ft = timing.time()
        print('_common_delay: Finished interpolation')
        print('Interpolation took {:4.2f} secs'.format(ft-st))
        print('Average of {:1.6f} secs/ray'.format(.5*(ft-st)/len(rays)))
        print('_common_delay: finished point-wise delay calculations')
        print('First ten points along last ray: {}'.format(ray[:10,:]))
        print('First ten points interpolated wet delay: {}'.format(wet_pw[-1][:10]))
        print('First ten points interpolated hydrostatic delay: {}'.format(hydro_pw[-1][:10]))
        print('New stepSize = {}'.format(stepSize))

    # intergrate the point-wise delays to get total slant delay
    delays = _integrateLOS(stepSize, wet_pw, hydro_pw)
 
    if verbose:
        print('Finished integration')
        print('First ten wet delay estimates: {}'.format(delays[0][:10]))
        print('First ten hydrostatic delay estimates: {}'.format(delays[1][:10]))

    return delays


def getIntFcn(weatherObj, itype = 'wet', interpType = 'scipy'):
    '''
    Function to create and return an Interpolator object
    '''
    from RAiDER.interpolator import Interpolator

    ifFun = Interpolator()
    ifFun.setPoints(*weatherObj.getPoints())
    ifFun.setProjection(weatherObj.getProjection())

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
    return 1e-6*np.trapz(pw, zs, axis = -1)


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


def reprojectRays(rays, oldProj, newProj):
    '''
    Reproject rays into the weather model projection, then rearrange to 
    match the weather model ordering. Rays are assumed to be in ECEF.
    '''
    from RAiDER.delayFcns import _transform
    newPts = [_transform(ray, oldProj, newProj) for ray in rays]
    return newPts


def orderForInterp(inRays):
    '''
    re-order a set of rays to match the interpolator ordering
    '''
    rays = []
    for pnt in inRays:
        rays.append(np.array([pnt[:,1], pnt[:,0], pnt[:,2]]).T)
    return rays


def tropo_delay(time, los = None, lats = None, lons = None, heights = None, 
                weather = None, wmFileLoc = None, zref = 15000, out = None, 
                parallel=True,verbose = False, download_only = False):
    """Calculate troposphere delay from command-line arguments.

    We do a little bit of preprocessing, then call
    _common_delay. 
    """
    from RAiDER.models.allowed import checkIfImplemented
    from RAiDER.downloadWM import downloadWMFile
    from RAiDER.llreader import getHeights 
    import RAiDER.util

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
    checkIfImplemented(weather_model_name.upper().replace('-',''))

    # check whether weather model files are supplied
    if weather_files is None:
        download_flag, f = downloadWMFile(weather_model.Model(), time, wmFileLoc, verbose)
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
        print('Weather Model Name: {}'.format(weather_model.Model()))
        print(weather_model)
        #p = weather.plot(p)

    # Pull the lat/lon data if using the weather model 
    if lats is None or len(lats)==2:
        uwn = True
        lats,lons = weather_model.getLL() 
        lla = weather_model.getProjection()
        RAiDER.util.writeLL(time, lats, lons,lla, weather_model_name, out)
    else:
        uwn = False
    
    # check for compatilibility of the weather model locations and the input
    if RAiDER.util.isOutside(RAiDER.util.getExtent(lats, lons), RAiDER.util.getExtent(*weather_model.getLL())):
        print('WARNING: some of the requested points are outside of the existing' +
             'weather model; these will end up as NaNs')
 
    # Pull the DEM
    if verbose: 
        print('Beginning DEM calculation')
    lats, lons, hgts = getHeights(lats, lons,heights)

    # LOS check and load
    if verbose: 
        print('Beginning LOS calculation')
    if los is None:
        los = Zenith
    elif los is Zenith:
        pass
    else:
        import RAiDER.losreader
        los = RAiDER.losreader.infer_los(los, lats, lons, hgts, zref)

    if los is Zenith:
        raytrace = False
    else:
        raytrace = True
       
    RAiDER.util.checkShapes(los, lats, lons, hgts)
    RAiDER.util.checkLOS(los, raytrace, np.prod(lats.shape))

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

    # If the verbose option is called, write out the weather model to a pickle file
    if verbose:
        print('Saving weather model object to pickle file')
        import pickle
        pickleFilename = os.path.join(out, 'pickledWeatherModel.pik')
        with open(pickleFilename, 'wb') as f:
           pickle.dump(weather_model, f)

    # Call _common_delay
    if verbose:
       print('Lats shape is {}'.format(lats.shape))
       print('lat/lon box is {}/{}/{}/{} (SNWE)'
              .format(np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)))
       print('Type of LOS is {}'.format([los if type(los) is type else type(los)][0]))
       print('Height range is {0:.2f}-{0:.2f} m'.format(np.nanmin(hgts), np.nanmax(hgts)))
       print('Reference z-value (max z for integration) is {} m'.format(zref))
       print('Number of processors to use: {}'.format(nproc))
       print('Using weather nodes only? (true/false): {}'.format(uwn))
       print('Weather model: {}'.format(weather_model.Model()))
       print('Number of weather model nodes: {}'.format(np.prod(weather_model.getWetRefractivity().shape)))
       print('Shape of weather model: {}'.format(weather_model.getWetRefractivity().shape))
       print('Bounds of the weather model: {}/{}/{}/{} (SNWE)'
              .format(np.nanmin(weather_model._ys), np.nanmax(weather_model._ys), 
                     np.nanmin(weather_model._xs), np.nanmax(weather_model._xs)))
       print('Mean value of the wet refractivity: {}'
              .format(np.nanmean(weather_model.getWetRefractivity())))
       print('Mean value of the hydrostatic refractivity: {}'
              .format(np.nanmean(weather_model.getHydroRefractivity())))

    wet, hydro = _common_delay(weather_model, lats, lons, hgts, los, zref = zref,
                  useWeatherNodes = uwn, nproc = nproc, useDask = useDask, verbose = verbose)
    if verbose: 
        print('Finished delay calculation')

    # Restore shape
    try:
        hydro, wet = np.stack((hydro, wet)).reshape((2,) + real_shape)
    except:
        pass

    import pdb; pdb.set_trace()
    return wet, hydro
