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

from RAiDER.constants import _STEP,_ZREF


def interpolateDelay(weather_model_file_name, pnts_file_name, 
                  zref = _ZREF, useWeatherNodes = False,
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
     heights    - Grid of heights for each ground point
     look_vecs  - Grid of look vectors streching from ground point to sensor (cut off at zref)
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
        st = timing.time()

    # Determine if/what type of parallization to use
    if useDask:
       import dask.bag as db
    elif nproc > 1:
       import multiprocessing as mp
    else:
       pass

    # If weather model nodes only are desired, the calculation is very quick
    if useWeatherNodes:
        _,_,zs = weatherObj.getPoints()
        wet_pw  = weatherObj.getWetRefractivity()
        hydro_pw= weatherObj.getHydroRefractivity()
        wet_delays = _integrateZenith(zs, wet_pw)
        hydro_delays = _integrateZenith(zs, hydro_pw)
        return wet_delays,hydro_delays

    if verbose:
        print('Beginning ray calculation')
        print('ZREF = {}'.format(zref))
        print('stepSize = {}'.format(stepSize))

    RAiDER.delayFcns.calculate_rays(pnts_file_name, stepSize, verbose = verbose)
    delays = RAiDER.delayFcns.get_delays(stepSize, pnts_file_name, 
                              weather_model_file_name, interpType = interpType, 
                              verbose = verbose)
    #rays = RAiDER.delayFcns.calculate_rays(pnts_file_name, stepSize = stepSize, verbose=verbose)

    #if verbose:
    #    print('First ten points along first ray: {}'.format(rays[0][:10,:]))
    #    print('First ten points along last ray: {}'.format(rays[-1][:10,:]))

    #wmProj = weatherObj.getProjection()
    #newRays = ecef2WM(rays, wmProj)
    #rays = orderForInterp(newRays)

    #if verbose:
    #    print('Finished ray calculation')
    #    ft = timing.time()
    #    print('Ray-tracing preliminaries took {:4.2f} secs'.format(ft-st))
    #    print('First ten points along first ray: {}'.format(rays[0][:10,:]))
    #    print('First ten points along last ray: {}'.format(rays[-1][:10,:]))
    #    try:
    #        print('NaN check: {}'.format(['PASSED' if np.sum(np.isnan(rays[0]))==0 else 'FAILED'][0]))
    #    except:
    #        print('Ray 1 has length 0')

    ## Define the interpolator objects
    #ifWet = getIntFcn(weatherObj,interpType =interpType)
    #ifHydro = getIntFcn(weatherObj,itype = 'hydro', interpType = interpType)

    #if verbose:
    #    print('Wet interpolator bounding box: {}'.format(ifWet._bbox))
    #    print('Hydrostatic interpolator bounding box: {}'.format(ifHydro._bbox))
    #    print('Beginning interpolation of each ray')
    #    st = timing.time()

    #if useDask:
    #    Npart = min(len(newPts)//100 + 1, 1000)
    #    PntBag = db.from_sequence(newPts, npartitions=Npart)
    #    wet_pw = PntBag.map(interpRay).compute()
    #    hydro_pw = PntBag.map(interpRay).compute()
    #elif nproc > 1:
    #    import multiprocessing as mp
    #    pool = mp.Pool(12)
    #    inp1 = zip([ifWet]*len(newPts), newPts)
    #    inp2 = zip([ifHydro]*len(newPts), newPts)

    #    wet_pw = pool.map(interpRay,inp1)
    #    hydro_pw = pool.map(interpRay, inp2)
    #else:
    #    wet_pw, hydro_pw = [], []
    #    count = 0
    #    for ray in rays:
    #        wet_pw.append(interpRay((ifWet, ray)))
    #        hydro_pw.append(interpRay((ifHydro, ray)))
    #        count = count+1

    #if verbose:
    #    ft = timing.time()
    #    print('interpolateDelay: Finished interpolation')
    #    print('Interpolation took {:4.2f} secs'.format(ft-st))
    #    print('Average of {:1.6f} secs/ray'.format(.5*(ft-st)/len(rays)))
    #    print('interpolateDelay: finished point-wise delay calculations')
    #    print('First ten points along last ray: {}'.format(ray[:10,:]))
    #    print('First ten points interpolated wet delay: {}'.format(wet_pw[-1][:10]))
    #    print('First ten points interpolated hydrostatic delay: {}'.format(hydro_pw[-1][:10]))
    #    print('New stepSize = {}'.format(stepSize))

    ## intergrate the point-wise delays to get total slant delay
    #delays = _integrateLOS(stepSize, wet_pw, hydro_pw)
 
    #if verbose:
    #    print('Finished integration')
    #    print('First ten wet delay estimates: {}'.format(delays[0][:10]))
    #    print('First ten hydrostatic delay estimates: {}'.format(delays[1][:10]))

    return delays


# call the interpolator on each ray
def interpRay(tup):
    fcn, ray = tup
    return fcn(ray)[0]


def _integrateZenith(zs, pw):
    return 1e-6*np.trapz(pw, zs, axis = -1)


 
def computeDelay(weather_model_file_name, pnts_file_name, 
                 zref = _ZREF, out = None, parallel=True,verbose = False):
    """Calculate troposphere delay from command-line arguments.

    We do a little bit of preprocessing, then call
    interpolateDelay. 
    """
    if verbose: 
        print('Beginning delay calculation')

    # Call interpolateDelay to compute the hydrostatic and wet delays
    if parallel:
        useDask = True
        nproc = 16
    else:
        useDask = False
        nproc = 1

    # Call interpolateDelay
    if verbose:
        print('Reference z-value (max z for integration) is {} m'.format(zref))
        print('Number of processors to use: {}'.format(nproc))

    wet, hydro = interpolateDelay(weather_model_file_name, pnts_file_name, zref = zref,
                  nproc = nproc, useDask = useDask, verbose = verbose)
    if verbose: 
        print('Finished delay calculation')

    return wet, hydro


def tropo_delay(los, lats, lons, heights, flag, weather_model, wmLoc, zref,
         outformat, time, out, download_only, parallel, verbose,
         wetFilename, hydroFilename):
    """
    raiderDelay main function.
    """
    from RAiDER.llreader import getHeights 
    from RAiDER.losreader import getLookVectors
    from RAiDER.processWM import prepareWeatherModel
    from RAiDER.utilFcns import writeDelays

    if verbose:
        print('Starting to run the weather model calculation')
        print('Time type: {}'.format(type(time)))
        print('Time: {}'.format(time.strftime('%Y%m%d')))
        print('Parallel is {}'.format(parallel))

    # location of the weather model files
    if verbose:
        print('Beginning weather model pre-processing')
        print('Download-only is {}'.format(download_only))
    if wmLoc is None:
        wmLoc = os.path.join(out, 'weather_files')

    # weather model calculation
    wm_filename = '{}_{}.h5'.format(weather_model['name'], time)
    weather_model_file = os.path.join(wmLoc, wm_filename)
    if not os.path.exists(weather_model_file):
        weather_model, lats, lons = prepareWeatherModel(weather_model, wmLoc, out, lats=lats,  
                        lons=lons, time=time, verbose=verbose, download_only=download_only)
        try:
            weather_model.write2HDF5(weather_model_file)
        except:
            pass
        del weather_model

    if download_only:
        return None, None

    pnts_file = os.path.join('geom', 'testx.h5')
    if not os.path.exists(pnts_file):
        # Pull the DEM.
        if verbose:
            print('Beginning DEM calculation')
        lats, lons, hgts = getHeights(lats, lons,heights)
        in_shape = lats.shape

        # Convert the line-of-sight inputs to look vectors
        if verbose:
            print('Lats shape is {}'.format(lats.shape))
            print('lat/lon box is {}/{}/{}/{} (SNWE)'
                   .format(np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)))
            print('DEM height range is {0:.2f}-{1:.2f} m'.format(np.nanmin(hgts), np.nanmax(hgts)))
            print('Beginning line-of-sight calculation')
        los = getLookVectors(los, lats, lons, hgts, zref)

        # write to an HDF5 file
        writePnts2HDF5(lats, lons, hgts, los, pnts_file, in_shape)
        del lats, lons, hgts, los

    wetDelay, hydroDelay = \
       computeDelay(weather_model_file, pnts_file, zref, out,
                         parallel=parallel, verbose = verbose)

    writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                outformat, wetFilename, hydroFilename,
                proj = None, gt = None, ndv = 0.)

    return wetDelay, hydroDelay


def writePnts2HDF5(lats, lons, hgts, los, outName = 'testx.h5', in_shape = None):
    '''
    Write query points to an HDF5 file for storage and access
    '''
    import datetime
    import h5py
    import os

    from RAiDER.utilFcns import checkLOS

    if in_shape is None:
        in_shape = llas.shape[:-1]
    checkLOS(los, np.prod(lats.shape))

    # Save the shape so we can restore later, but flatten to make it
    # easier to think about
    llas = np.stack((lats, lons, hgts), axis=-1)
    llas = llas.reshape(-1, 3)
    lats, lons, hgts = np.moveaxis(llas, -1, 0)
    los = los.reshape((np.prod(los.shape[:-1]), los.shape[-1]))

    with h5py.File(outName, 'w') as f:
    #with h5py.File(outName, 'w', chunk_cache_mem_size=1024**2*4000) as f:
        x = f.create_dataset('lon', data = lons.flatten(), chunks = True)
        y = f.create_dataset('lat', data = lats.flatten(), chunks = x.chunks)
        z = f.create_dataset('hgt', data = hgts.flatten(), chunks = x.chunks)
        los = f.create_dataset('LOS', data= los, chunks = x.chunks + (3,))
        x.attrs['Shape'] = in_shape
        y.attrs['Shape'] = in_shape
        z.attrs['Shape'] = in_shape

        start_positions = f.create_dataset('Rays_SP', (len(x),3), chunks = los.chunks)
        lengths = f.create_dataset('Rays_len',  (len(x),), chunks = x.chunks)
        lengths.attrs['NumRays'] = len(x)
        scaled_look_vecs = f.create_dataset('Rays_SLV',  (len(x),3), chunks = los.chunks)
     for band in range(ds.RasterCount):
@@ -178,7 +179,7 @@ def gdal_open(fname, returnProj = False):
     if not returnProj:
         return data
     else:
-        return data, proj
+        return data, proj, gt

