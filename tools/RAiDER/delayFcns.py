#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
Calculated rays from the ground to the top of the atmosphere
(or more precisely, to a reference height zref). 

Currently we take samples every _STEP meters.
"""

# standard imports
import numpy as np
import os
import pyproj

# local imports
import constants as const
from constants import Zenith
import util

# Step in meters to use when integrating
_STEP = const._STEP
# Default top of the atmosphere
_ZREF = const._ZMAX


def _getZenithLookVecs(lats, lons, heights, zref = _ZREF):

    '''
    Returns look vectors when Zenith is used. 
    Inputs: 
       lats/lons/heights - Nx1 numpy arrays of points. 
       zref              - float, integration height in meters
    Outputs: 
       zenLookVecs       - an Nx3 numpy array with the look vectors.
                           The vectors give the zenith ray paths for 
                           each of the points to the top of the atmosphere. 
    '''
    zenLookVecs = (np.array((util.cosd(lats)*util.cosd(lons),
                              util.cosd(lats)*util.sind(lons),
                              util.sind(lats))).T
                    * (zref - heights)[..., np.newaxis])
    return zenLookVecs


def _get_lengths(look_vecs):
    '''
    Returns the lengths of a vector or set of vectors, fast. 
    Inputs: 
       looks_vecs  - an Nx3 numpy array containing look vectors with absolute
                     lengths; i.e., the absolute position of the top of the
                     atmosphere. 
    Outputs: 
       lengths     - an Nx1 numpy array containing the absolute distance in 
                     meters of the top of the atmosphere from the ground pnt. 
    '''
    lengths = np.linalg.norm(look_vecs, axis=-1)
    lengths[~np.isfinite(lengths)] = 0
    return lengths


def _ray_helper(tup):
    from get_rays import compute_ray as _compute_ray
    return _compute_ray(tup[0], tup[1], tup[2], tup[3])


def _get_rays(lengths, stepSize, start_positions, scaled_look_vecs, Nproc = None):
    '''
    Create the integration points for each ray path. 
    '''
    data = zip(lengths, start_positions, scaled_look_vecs, [stepSize]*len(lengths))
    if len(lengths)<1e6:
       positions_l= []
       for tup in rayData:
           positions_l.append(_ray_helper(tup))
    else:
       import multiprocessing as mp
       if Nproc is None:
          Nproc =mp.cpu_count()*3//4
       pool = mp.Pool()
       positions_l = pool.map(_ray_helper, data)

    return positions_l


def _transform(ray, oldProj, newProj):
    '''
    Transform a ray from one coordinate system to another
    '''
    newRay = np.stack(
                pyproj.transform(
                      oldProj, newProj, ray[:,0], ray[:,1], ray[:,2])
                      ,axis = -1)
    return newRay


def _re_project(tup): 
    newPnt = _transform(tup[0],tup[1], tup[2])
    return newPnt
def f(x):
    ecef = pyproj.Proj(proj='geocent')
    return _transform(x, ecef, newProj)

def sortSP(arr):
    '''
    Return an array that has been sorted progressively by 
    each axis, beginning with the first
    Input:
      arr  - an Nx(2 or 3) array containing a set of N points in 2D or 3D space
    Output:
      xSorted  - an Nx(2 or 3) array containing the sorted points
    '''
    ySorted = arr[arr[:,1].argsort()]
    xSorted = ySorted[ySorted[:,0].argsort()]
    return xSorted


def reproject(inlat, inlon, inhgt, inProj, outProj):
    '''
    reproject a set of lat/lon/hgts to a new coordinate system
    '''
    import pyproj
    return np.array(pyproj.transform(inProj, outProj, lon, lat, height)).T


def calculate_rays(lats, lons, heights, look_vecs = Zenith, zref = None, setupSize = _STEP, verbose = False):
    '''
    From a set of lats/lons/hgts, compute ray paths from the ground to the 
    top of the atmosphere, using either a set of look vectors or the zenith
    '''
    if verbose:
        print('calculate_rays: Starting look vector calculation')
        print('The integration stepsize is {} m'.format(stepSize))
    
    if look_vecs is Zenith:
        look_vecs = _getZenithLookVecs(lats, lons, heights, zref = zref)
        lengths = np.array([_ZREF]*len(look_vecs))
    else:
        # Otherwise, set off on the interpolation road
        mask = np.isnan(heights)
    
        # Get the integration points along the look vectors
        # First get the length of each look vector, get integration steps along 
        # each, then get the unit vector pointing in the same direction
        lengths = _get_lengths(look_vecs)
        lengths[mask] = np.nan

    scaled_look_vecs = look_vecs / lengths[..., np.newaxis]


    # TODO: make lla an optional input to the fcn? 
    ecef = pyproj.Proj(proj='geocent')
    lla = pyproj.Proj(proj='latlong')
    # this calculation takes a long time
    start_positions = reproject(lats, lons, heights, lla, ecef)
    # TODO: Check that start_positions is an array of Nx3 size
    start_positions = sortSP(start_positions)

    # cythonize these lines?
    positions_l= _get_rays(lengths, stepSize, start_positions, scaled_look_vecs)
    
    if verbose:
        print('_common_delay: Finished _get_rays')
        ft = time.time()
        print('Ray initialization took {:4.2f} secs'.format(ft-st))
        print('_common_delay: Starting _re_project')
        st = time.time()

     TODO: Problem: This part could be parallelized, but Dask is slow. 
        # perhaps should use multiprocessing or cythonize some fcns?
        newProj = weatherObj.getProjection()
        if useDask:
            if verbose:
                print('Beginning re-projection using Dask')
            Npart = min(len(positions_l)//100 + 1, 1000)
            bag = [(pos, ecef, newProj) for pos in positions_l]
            PntBag = db.from_sequence(bag, npartitions=Npart)
            newPts = PntBag.map(_re_project).compute()
        else:
            if verbose:
                print('Beginning re-projection without Dask')
            newPts = list(map(f, positions_l))

        # TODO: not sure how long this takes but looks inefficient
        newPts = [np.vstack([p[:,1], p[:,0], p[:,2]]).T for p in newPts]

        if verbose:
            print('_common_delay: Finished re-projecting')
            print('_common_delay: The size of look_vecs is {}'.format(np.shape(look_vecs)))
            ft = time.time()
            print('Re-projecting took {:4.2f} secs'.format(ft-st))
            print('_common_delay: Starting Interpolation')
            st = time.time()     


    # chunk the rays
    rayChunkIndices = list(range(len(start_positions)))
    chunks = np.array_split(rayChunkIndices, nChunks)
    bags = []
    for chunk in chunks:
       bags.append(newPts[chunk])

