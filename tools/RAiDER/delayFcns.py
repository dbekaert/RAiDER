#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import h5py
import numpy as np
import pyproj

from RAiDER.constants import _STEP
from RAiDER import makePoints

def makePoints3D(max_len, Rays_SP, Rays_SLV, stepSize):
    '''
    Python version of cython code to create the rays needed for ray-tracing
    Inputs: 
      max_len: maximum length of the rays
      Rays_SP: Nx x Ny x Nz x 3 numpy array of the location of the ground pixels in an earth-centered, 
               earth-fixed coordinate system
      Rays_SLV: Nx x Ny x Nz x 3 numpy array of the look vectors pointing from the ground pixel to the sensor
      stepSize: Distance between points along the ray-path
    Output:
      ray: a Nx x Ny x Nz x 3 x Npts array containing the rays tracing a path from the ground pixels, along the 
           line-of-sight vectors, up to the maximum length specified.
    '''
    Npts  = int((max_len+stepSize)//stepSize)
    nrow = Rays_SP.shape[0]
    ncol = Rays_SP.shape[1]
    nz = Rays_SP.shape[2]
    ray = np.empty((nrow, ncol, nz, 3, Npts), dtype=np.float64)
    basespace = np.arange(0, max_len, stepSize) # max_len+stepSize

    for k1 in range(nrow):
        for k2 in range(ncol):
            for k2a in range(nz):
                for k3 in range(3):
                        ray[k1,k2,k2a,k3,:] = Rays_SP[k1,k2,k2a,k3] + basespace*Rays_SLV[k1,k2,k2a,k3]
    return ray

def _ray_helper(lengths, start_positions, scaled_look_vectors, stepSize):
    #return _compute_ray(tup[0], tup[1], tup[2], tup[3])
    maxLen = np.nanmax(lengths)
    out, rayLens = [],[]
    for L, S, V, in zip(lengths, start_positions, scaled_look_vectors):
        ray, Npts = _compute_ray(L, S, V, stepSize, maxLen)
        out.append(ray)
        rayLens.append(Npts)
    return np.stack(out, axis = 0), np.array(rayLens)


def _compute_ray2(L, S, V, stepSize):
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

def _compute_ray(L, S, V, stepSize, maxLen):
    '''
    Compute and return points along a ray, given a total length, 
    start position (in x,y,z), a unit look vector V, and the 
    stepSize.
    '''
    thisspace = np.arange(0, maxLen+stepSize, stepSize)
    ray = S + thisspace[..., np.newaxis]*V
    return ray, int(L//stepSize + 1)


def testTime(t, ray, N):
    import time
    st = time.time()
    for k in range(N):
        ray_x, ray_y, ray_z = t.transform(ray[...,0], ray[...,1], ray[...,2])
    et = time.time()
    return (et - st)/N


def get_delays(stepSize, pnts_file, wm_file, interpType = '3D', verbose = False):
    '''
    Create the integration points for each ray path. 
    '''
    from RAiDER.delayFcns import _transform 
    from pyproj import Transformer, CRS 
    from tqdm import tqdm

    # Transformer from ECEF to weather model
    p1 = CRS.from_epsg(4978) 
    proj_wm = getProjFromWMFile(wm_file)
    t = Transformer.from_proj(p1,proj_wm) 

    # Get the weather model data
    with h5py.File(wm_file, 'r') as f:
        xs_wm = f['x'].value.copy()
        ys_wm = f['y'].value.copy()
        zs_wm = f['z'].value.copy()
        wet=f['wet'].value.copy()
        hydro=f['hydro'].value.copy()

    ifWet = getIntFcn(xs_wm, ys_wm, zs_wm, wet)
    ifHydro = getIntFcn(xs_wm, ys_wm, zs_wm, hydro)

    delays = []
    with h5py.File(pnts_file, 'r') as f:
        Nrays = f['Rays_len'].attrs['NumRays']
        chunkSize = f['lon'].chunks[0]
        in_shape = f['lon'].attrs['Shape']

    fac = 10
    chunkSize = chunkSize//fac
    Nchunks = Nrays//chunkSize + 1

    with h5py.File(pnts_file, 'r') as f:
        for k in tqdm(range(Nchunks)):
        #for index in tqdm(range(Nrays)):
            index = np.arange(k*chunkSize, min((k+1)*chunkSize, Nrays))
            
            Npts = [int(L//stepSize + 1) for L in f['Rays_len'][index]]
#            ray, Npts = _ray_helper(f['Rays_len'][index], 
#                                    f['Rays_SP'][index,:], 
#                                    f['Rays_SLV'][index,:], stepSize)
            ray = makePoints(max_len, f['Rays_SP'][index,:].value.copy(), f['Rays_SLV'][index,:],stepSize)
            #if f['Rays_len'][index] > 1:
            #    ray = _compute_ray2(f['Rays_len'][index], 
            #                            f['Rays_SP'][index,:], 
            #                            f['Rays_SLV'][index,:], stepSize)
            ray_x, ray_y, ray_z = t.transform(ray[...,0], ray[...,1], ray[...,2])
            delay_wet   = interpolate2(ifWet, ray_x, ray_y, ray_z)
            delay_hydro = interpolate2(ifHydro, ray_x, ray_y, ray_z)
            delays.append(_integrateLOS(stepSize, delay_wet, delay_hydro, Npts))
            #else:
            #    delays.append(np.array(np.nan, np.nan))

#    wet_delay, hydro_delay = [], []
#    for d in delays:
#        wet_delay.append(d[0,...])
#        hydro_delay.append(d[1,...])

    wet_delay = np.concatenate([d[0,...] for d in delays]).reshape(in_shape)
    hydro_delay = np.concatenate([d[1,...] for d in delays]).reshape(in_shape)

    # Restore shape
#    try:
#        hydro, wet = np.stack((hydro_delay, wet_delay)).reshape((2,) + real_shape)
#    except:
#        pass

#    return wet, hydro
    return wet_delay, hydro_delay


def interpolate2(fun, x, y, z):
    '''
    helper function to make the interpolation step cleaner
    '''
    in_shape = x.shape
    out = fun((y.flatten(), x.flatten(), z.flatten()))
    outData = out.reshape(in_shape)
    return outData

def interpolate(fun, x, y, z):
    '''
    helper function to make the interpolation step cleaner
    '''
    in_shape = x.shape
    out = []
    flat_shape = np.prod(in_shape)
    for x, y, z in zip(y.reshape(flat_shape,), x.reshape(flat_shape,), z.reshape(flat_shape,)):
        out.append(fun((y,x,z))) # note that this re-ordering is on purpose to match the weather model
    outData = np.array(out).reshape(in_shape)
    return outData


def _integrateLOS(stepSize, wet_pw, hydro_pw, Npts = None):
    delays = [] 
    for d in (wet_pw, hydro_pw):
        delays.append(_integrate_delays(stepSize, d, Npts))
    return np.stack(delays, axis = 0)

def _integrate_delays2(stepSize, refr):
    '''
    This function gets the actual delays by integrating the refractivity in 
    each node. Refractivity is given in the 'refr' variable. 
    '''
    return int_fcn2(refr, stepSize)

def _integrate_delays(stepSize, refr, Npts):
    '''
    This function gets the actual delays by integrating the refractivity in 
    each node. Refractivity is given in the 'refr' variable. 
    '''
    delays = []
    for n, ray in zip(Npts, refr):
        delays.append(int_fcn(ray, stepSize, n))
    return np.array(delays)

def int_fcn(y, dx, N):
    return 1e-6*dx*np.nansum(y[:N])
def int_fcn2(y, dx):
    return 1e-6*dx*np.nansum(y)

def getIntFcn(xs, ys, zs, var):
    '''
    Function to create and return an Interpolator object
    '''
    from scipy.interpolate import RegularGridInterpolator as rgi
    ifFun = rgi((ys.flatten(),xs.flatten(), zs.flatten()), var,bounds_error=False, fill_value = np.nan)
    return ifFun

def getProjFromWMFile(wm_file):
    '''
    Returns the projection of an HDF5 file 
    '''
    from pyproj import CRS 
    with h5py.File(wm_file, 'r') as f:
        wm_proj = CRS.from_json(f['Projection'].value)
    return wm_proj


def _transform(ray, oldProj, newProj):
    '''
    Transform a ray from one coordinate system to another
    '''
    newRay = np.stack(
                pyproj.transform(
                      oldProj, newProj, ray[:,0], ray[:,1], ray[:,2])
                      ,axis = -1, always_xy = True)
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


def lla2ecef(pnts_file):
    '''
    reproject a set of lat/lon/hgts to a new coordinate system
    '''
    from pyproj import Transformer

    t = Transformer.from_crs(4326,4978) # converts from WGS84 geodetic to WGS84 geocentric
    with h5py.File(pnts_file, 'r+') as f:
        f['Rays_SP'][:,:] = np.array(t.transform(f['lon'].value, f['lat'].value, f['hgt'].value)).T


def getUnitLVs(pnts_file):
    '''
    Get a set of look vectors normalized by their lengths
    '''
    get_lengths(pnts_file)
    with h5py.File(pnts_file, 'r+') as f:
        f['Rays_SLV'][:,:] = f['LOS'].value / f['Rays_len'].value[:,np.newaxis]


def get_lengths(pnts_file):
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
    with h5py.File(pnts_file, 'r+') as f:
        lengths = np.linalg.norm(f['LOS'].value, axis=-1)
        try:
            lengths[~np.isfinite(lengths)] = 0
        except TypeError:
            if ~np.isfinite(lengths):
                lengths = 0
        f['Rays_len'][:] = lengths
        f['Rays_len'].attrs['MaxLen'] = np.nanmax(lengths)



def calculate_rays(pnts_file, stepSize = _STEP, verbose = False):
    '''
    From a set of lats/lons/hgts, compute ray paths from the ground to the 
    top of the atmosphere, using either a set of look vectors or the zenith
    '''
    if verbose:
        print('calculate_rays: Starting look vector calculation')
        print('The integration stepsize is {} m'.format(stepSize))

    # get the lengths of each ray for doing the interpolation 
    getUnitLVs(pnts_file) 

    # This projects the ground pixels into earth-centered, earth-fixed coordinate 
    # system and sorts by position
    newPts = lla2ecef(pnts_file)

    # This returns the list of rays
    # TODO: make this not a list. 
    # Why is a list used instead of a numpy array? It is because every ray has a
    # different length, and some rays have zero length (i.e. the points over
    # water). However, it would be MUCH more efficient to do this as a single 
    # pyproj call, rather than having to send each ray individually. For right 
    # now we bite the bullet.
    #TODO: write the variables to a chunked HDF5 file and then use this file
    # to compute the rays. Write out the rays to the file. 
    #TODO: Add these variables to the original HDF5 file so that we don't have
    # to create a new file

#    return delays 


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
