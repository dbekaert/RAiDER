#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pyproj

from RAiDER.constants import _STEP


def _ray_helper(tup):
    return _compute_ray(tup[0], tup[1], tup[2], tup[3])


def _compute_ray(L, S, V, stepSize):
    '''
    Compute and return points along a ray, given a total length, 
    start position (in x,y,z), a unit look vector V, and the 
    stepSize.
    '''
    # Have to handle the case where there are invalid data
    # TODO: cythonize this? 
    try:
        thisspace = np.arange(0, L+stepSize, stepSize)
    except ValueError:
        thisspace = np.array([])
    ray = S + thisspace[..., np.newaxis]*V
    return ray


def _get_rays(lengths, stepSize, start_positions, scaled_look_vecs, Nproc = None):
    '''
    Create the integration points for each ray path. 
    '''
    data = zip(lengths, start_positions, scaled_look_vecs, [stepSize]*len(lengths))
    if len(lengths)<1e6:
       positions_l= []
       for tup in data:
           positions_l.append(_ray_helper(tup))
    else:
       import multiprocessing as mp
       if Nproc is None:
          Nproc =mp.cpu_count()*3//4
       pool = mp.Pool(Nproc)
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


def lla2ecef(inlat, inlon, inhgt):
    '''
    reproject a set of lat/lon/hgts to a new coordinate system
    '''
    from pyproj import Transformer
    t = Transformer.from_crs(4326,4978) # converts from WGS84 geodetic to WGS84 geocentric
    return np.array(t.transform(inlon, inlat, inhgt)).T


def getUnitLVs(look_vecs):
    '''
    Return a set of look vectors normalized by their lengths
    '''
    lengths = _get_lengths(look_vecs)
    slvs = look_vecs / lengths[..., np.newaxis]
    return slvs,lengths


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
    if look_vecs.ndim==1:
       if len(look_vecs)!=3:
          raise RuntimeError('look_vecs must be Nx3') 
    if look_vecs.shape[-1]!=3:
       raise RuntimeError('look_vecs must be Nx3')

    lengths = np.linalg.norm(look_vecs, axis=-1)
    try:
        lengths[~np.isfinite(lengths)] = 0
    except TypeError:
        if ~np.isfinite(lengths):
            lengths = 0

    return lengths


def calculate_rays(lats, lons, heights, look_vecs, stepSize = _STEP, verbose = False):
    '''
    From a set of lats/lons/hgts, compute ray paths from the ground to the 
    top of the atmosphere, using either a set of look vectors or the zenith
    '''
    if verbose:
        print('calculate_rays: Starting look vector calculation')
        print('The integration stepsize is {} m'.format(stepSize))

    # get the lengths of each ray for doing the interpolation 
    scaled_look_vecs, lengths = getUnitLVs(look_vecs) 

    # This projects the ground pixels into earth-centered, earth-fixed coordinate 
    # system and sorts by position
    start_positions = lla2ecef(lats, lons, heights)

    # This returns the list of rays
    # TODO: make this not a list. 
    # Why is a list used instead of a numpy array? It is because every ray has a
    # different length, and some rays have zero length (i.e. the points over
    # water). However, it would be MUCH more efficient to do this as a single 
    # pyproj call, rather than having to send each ray individually. For right 
    # now we bite the bullet.
    rays = _get_rays(lengths, stepSize, start_positions, scaled_look_vecs)

    return rays


#    # Now to interpolate, we have to re-project each ray into the coordinate 
#    # system used by the weather model.  --> this is inefficient as is
#    if useDask:
#        if verbose:
#            print('Beginning re-projection using Dask')
#        Npart = min(len(positions_l)//100 + 1, 1000)
#        bag = [(pos, ecef, newProj) for pos in positions_l]
#        PntBag = db.from_sequence(bag, npartitions=Npart)
#        newPts = PntBag.map(_re_project).compute()
#    else:
#        if verbose:
#            print('Beginning re-projection without Dask')
#        newPts = list(map(f, positions_l))
#
#    # TODO: not sure how long this takes but looks inefficient
#    newPts = [np.vstack([p[:,1], p[:,0], p[:,2]]).T for p in newPts]


     # TODO: implement chunking for efficient interpolation?
#    # chunk the rays
#    rayChunkIndices = list(range(len(start_positions)))
#    chunks = np.array_split(rayChunkIndices, nChunks)
#    bags = []
#    for chunk in chunks:
#       bags.append(newPts[chunk])
#
