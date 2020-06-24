#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson, David Bekaert & Yang Lei
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import multiprocessing as mp
import time

import h5py
import numpy as np
import pyproj

from RAiDER.constants import _STEP
from RAiDER.makePoints import makePoints1D


def _ray_helper(lengths, start_positions, scaled_look_vectors, stepSize):
    # return _compute_ray(tup[0], tup[1], tup[2], tup[3])
    maxLen = np.nanmax(lengths)
    out, rayLens = [], []
    for L, S, V, in zip(lengths, start_positions, scaled_look_vectors):
        ray, Npts = _compute_ray(L, S, V, stepSize, maxLen)
        out.append(ray)
        rayLens.append(Npts)
    return np.stack(out, axis=0), np.array(rayLens)


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


def get_delays(stepSize, pnts_file, wm_file, interpType='3D',
               verbose=False, delayType="Zenith", cpu_num=0):
    '''
    Create the integration points for each ray path.
    '''

    t0 = time.time()

    # Get the weather model data
    with h5py.File(wm_file, 'r') as f:
        xs_wm = f['x'][()].copy()
        ys_wm = f['y'][()].copy()
        zs_wm = f['z'][()].copy()
        wet = f['wet'][()].copy()
        hydro = f['hydro'][()].copy()

    ifWet = getIntFcn(xs_wm, ys_wm, zs_wm, wet)
    ifHydro = getIntFcn(xs_wm, ys_wm, zs_wm, hydro)

    with h5py.File(pnts_file, 'r') as f:
        Nrays = f.attrs['NumRays']
        chunkSize = f['lon'].chunks
        in_shape = f['lon'].attrs['Shape']
        arrSize = f['lon'].shape
        max_len = np.nanmax(f['Rays_len']).astype(np.float64)

    if cpu_num == 0:
        Nchunks = mp.cpu_count()
        cpu_num = Nchunks
    else:
        Nchunks = cpu_num

    chunkSize1 = int(np.floor(np.prod(in_shape) / Nchunks))
    chunkRem = np.prod(in_shape) - chunkSize1 * Nchunks

    with h5py.File(pnts_file, 'r') as f:
        CHUNKS = []
        for k in range(Nchunks):
            if k == (Nchunks-1):
                chunkInds = range(k*chunkSize1, (k+1)*chunkSize1+chunkRem)
            else:
                chunkInds = range(k*chunkSize1, (k+1)*chunkSize1)
            CHUNKS.append(chunkInds)
        chunk_inputs = [(kk, CHUNKS[kk], np.array(f['Rays_SP']), np.array(f['Rays_SLV']), in_shape, stepSize, ifWet, ifHydro, max_len, wm_file)
                        for kk in range(Nchunks)]
        with mp.Pool() as pool:
            individual_results = pool.map(unpacking_hdf5_read, chunk_inputs)

        delays = np.concatenate(individual_results)

    wet_delay = np.concatenate([d[0, ...] for d in delays]).reshape(in_shape)
    hydro_delay = np.concatenate([d[1, ...] for d in delays]).reshape(in_shape)

    time_elapse = (time.time() - t0)
    with open('get_delays_time_elapse.txt', 'w') as f:
        f.write('{}'.format(time_elapse))
    if verbose:
        time_elapse_hr = int(np.floor(time_elapse/3600.0))
        time_elapse_min = int(np.floor((time_elapse - time_elapse_hr*3600.0)/60.0))
        time_elapse_sec = (time_elapse - time_elapse_hr*3600.0 - time_elapse_min*60.0)
        print("Delay estimation cost {0} hour(s) {1} minute(s) {2} second(s) using {3} cpu threads".format(time_elapse_hr, time_elapse_min, time_elapse_sec, cpu_num))
    return wet_delay, hydro_delay


def unpacking_hdf5_read(tup):
    """
        Like numpy.apply_along_axis(), but and with arguments in a tuple
        instead.
        This function is useful with multiprocessing.Pool().map(): (1)
        map() only handles functions that take a single argument, and (2)
        this function can generally be imported from a module, as required
        by map().
        """

    from pyproj import Transformer, CRS

    k, chunkInds, SP, SLV, in_shape, stepSize, ifWet, ifHydro, max_len, wm_file = tup

    # Transformer from ECEF to weather model
    p1 = CRS.from_epsg(4978)
    proj_wm = getProjFromWMFile(wm_file)
    t = Transformer.from_proj(p1, proj_wm, always_xy=True)

    delays = []

    for ind in chunkInds:
        row, col = [v[0] for v in np.unravel_index([ind], in_shape)]
        ray = makePoints1D(max_len, SP[row, col, :].astype('float64'),
                           SLV[row, col, :].astype('float64'), stepSize)
        ray_x, ray_y, ray_z = t.transform(ray[..., 0, :], ray[..., 1, :], ray[..., 2, :])
        delay_wet   = interpolate2(ifWet, ray_x, ray_y, ray_z)
        delay_hydro = interpolate2(ifHydro, ray_x, ray_y, ray_z)
        delays.append(_integrateLOS(stepSize, delay_wet, delay_hydro))

    return delays


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
        out.append(fun((y, x, z)))  # note that this re-ordering is on purpose to match the weather model
    outData = np.array(out).reshape(in_shape)
    return outData


def _integrateLOS(stepSize, wet_pw, hydro_pw, Npts=None):
    delays = []
    for d in (wet_pw, hydro_pw):
        if d.ndim == 1:
            delays.append(np.array([int_fcn2(d, stepSize)]))
        else:
            delays.append(_integrate_delays(stepSize, d, Npts))
    return np.stack(delays, axis=0)


def _integrate_delays2(stepSize, refr):
    '''
    This function gets the actual delays by integrating the refractivity in
    each node. Refractivity is given in the 'refr' variable.
    '''
    return int_fcn2(refr, stepSize)


def _integrate_delays(stepSize, refr, Npts=None):
    '''
    This function gets the actual delays by integrating the refractivity in
    each node. Refractivity is given in the 'refr' variable.
    '''
    delays = []
    if Npts is not None:
        for n, ray in zip(Npts, refr):
            delays.append(int_fcn(ray, stepSize, n))
    else:
        for ray in refr:
            delays.append(int_fcn2(ray, stepSize))
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
    ifFun = rgi((ys.flatten(), xs.flatten(), zs.flatten()), var, bounds_error=False, fill_value=np.nan)
    return ifFun


def getProjFromWMFile(wm_file):
    '''
    Returns the projection of an HDF5 file
    '''
    from pyproj import CRS
    with h5py.File(wm_file, 'r') as f:
        wm_proj = CRS.from_json(f['Projection'][()])
    return wm_proj


def _transform(ray, oldProj, newProj):
    '''
    Transform a ray from one coordinate system to another
    '''
    newRay = np.stack(
                pyproj.transform(
                      oldProj, newProj, ray[:, 0], ray[:, 1], ray[:, 2])
                      , axis=-1, always_xy=True)
    return newRay


def _re_project(tup):
    newPnt = _transform(tup[0], tup[1], tup[2])
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
    ySorted = arr[arr[:, 1].argsort()]
    xSorted = ySorted[ySorted[:, 0].argsort()]
    return xSorted


def lla2ecef(pnts_file):
    '''
    reproject a set of lat/lon/hgts to a new coordinate system
    '''
    from pyproj import Transformer

    t = Transformer.from_crs(4326, 4978, always_xy=True)  # converts from WGS84 geodetic to WGS84 geocentric
    with h5py.File(pnts_file, 'r+') as f:
        sp = np.moveaxis(np.array(t.transform(f['lon'][()], f['lat'][()], f['hgt'][()])), 0, -1)
        f['Rays_SP'][...] = sp.astype(np.float64)  # ensure double is maintained


def getUnitLVs(pnts_file):
    '''
    Get a set of look vectors normalized by their lengths
    '''
    get_lengths(pnts_file)
    with h5py.File(pnts_file, 'r+') as f:
        slv = f['LOS'][()] / f['Rays_len'][()][..., np.newaxis]
        f['Rays_SLV'][...] = slv


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
        lengths = np.linalg.norm(f['LOS'][()], axis=-1)
        try:
            lengths[~np.isfinite(lengths)] = 0
        except TypeError:
            if ~np.isfinite(lengths):
                lengths = 0
        f['Rays_len'][:] = lengths.astype(np.float64)
        f['Rays_len'].attrs['MaxLen'] = np.nanmax(lengths)


def calculate_rays(pnts_file, stepSize=_STEP, verbose=False):
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
    # TODO: write the variables to a chunked HDF5 file and then use this file
    # to compute the rays. Write out the rays to the file.
    # TODO: Add these variables to the original HDF5 file so that we don't have
    # to create a new file
