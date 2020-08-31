#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson, David Bekaert & Yang Lei
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import itertools
import logging
import multiprocessing as mp
import time

import h5py
import numpy as np
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

from RAiDER.constants import _STEP, _ZREF
from RAiDER.interpolator import RegularGridInterpolator as Interpolator
from RAiDER.makeRays import makeRays1D

log = logging.getLogger(__name__)


def get_delays(stepSize, pnts_file, wm_file, interpType='3D', zref=_ZREF, cpu_num=0):
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

    ifWet = Interpolator((ys_wm, xs_wm, zs_wm), wet, fill_value=np.nan)
    ifHydro = Interpolator((ys_wm, xs_wm, zs_wm), hydro, fill_value=np.nan)

    with h5py.File(pnts_file, 'r') as f:
        chunkSize = f.attrs['ChunkSize']
        in_shape = f['lon'].attrs['Shape']

    CHUNKS = chunk(chunkSize, in_shape)
    Nchunks = len(CHUNKS)

    with h5py.File(pnts_file, 'r') as f:
        chunk_inputs = [(kk, CHUNKS[kk], np.array(f['Ray_start']), np.array(f['LOS']),
                         chunkSize, stepSize, ifWet, ifHydro, zref, wm_file) for kk in range(Nchunks)]

        with mp.Pool() as pool:
            individual_results = pool.starmap(process_chunk, chunk_inputs)
        delays = np.concatenate(individual_results)

    wet_delay = delays[0, ...].reshape(in_shape)
    hydro_delay = delays[1, ...].reshape(in_shape)

    time_elapse = (time.time() - t0)
    with open('get_delays_time_elapse.txt', 'w') as f:
        f.write('{}'.format(time_elapse))
    time_elapse_hr = int(np.floor(time_elapse / 3600.0))
    time_elapse_min = int(np.floor((time_elapse - time_elapse_hr * 3600.0) / 60.0))
    time_elapse_sec = (time_elapse - time_elapse_hr * 3600.0 - time_elapse_min * 60.0)
    log.debug(
        "Delay estimation cost %d hour(s) %d minute(s) %d second(s) using %d cpu threads",
        time_elapse_hr, time_elapse_min, time_elapse_sec, cpu_num
    )
    return wet_delay, hydro_delay


def make_interpolator(xs, ys, zs, data):
    '''
    Function to create and return an Interpolator object
    '''
    return RegularGridInterpolator(
        (ys.ravel(), xs.ravel(), zs.ravel()),
        data,
        bounds_error=False,
        fill_value=np.nan
    )


def chunk(chunkSize, in_shape):
    '''
    Create a set of indices to use as chunks
    '''
    startInds = makeChunkStartInds(chunkSize, in_shape)
    chunkInds = makeChunksFromInds(startInds, chunkSize, in_shape)
    return chunkInds


def makeChunkStartInds(chunkSize, in_shape):
    '''
    Create a list of start indices for chunking a numpy D-dimensional array.
    Inputs:
        chunkSize - length-D tuple containing chunk sizes
        in_shape  - length-D tuple containing the shape of the array to be chunked
    Outputs
        chunkInds - a list of length-D tuples, where each tuple is the starting
                    multi-index of each chunk
    Example:
        makeChunkStartInds((2,2,16), (4,8,16))
    Output:  [(0, 0, 0),
             (0, 2, 0),
             (0, 4, 0),
             (0, 6, 0),
             (2, 0, 0),
             (2, 2, 0),
             (2, 4, 0),
             (2, 6, 0)]
    '''
    if len(in_shape) == 1:
        chunkInds = [(i,) for i in range(0, in_shape[0], chunkSize[0])]

    elif len(in_shape) == 2:
        chunkInds = [(i, j) for i, j in itertools.product(range(0, in_shape[0], chunkSize[0]),
                                                          range(0, in_shape[1], chunkSize[1]))]
    elif len(in_shape) == 3:
        chunkInds = [(i, j, k) for i, j, k in itertools.product(range(0, in_shape[0], chunkSize[0]),
                                                                range(0, in_shape[1], chunkSize[1]),
                                                                range(0, in_shape[2], chunkSize[2]))]
    else:
        raise NotImplementedError('makeChunkStartInds: ndim > 3 not supported')

    return chunkInds


def makeChunksFromInds(startInd, chunkSize, in_shape):
    '''
    From a length-N list of tuples containing starting indices,
    create a list of indices into chunks of a numpy D-dimensional array.
    Inputs:
       startInd  - A length-N list of D-dimensional tuples containing the
                   starting indices of a set of chunks
       chunkSize - A D-dimensional tuple containing chunk size in each dimension
       in_shape  - A D-dimensional tuple containing the size of each dimension
    Outputs:
       chunks    - A length-N list of length-D lists, where each element of the
                   length-D list is a numpy array of indices
    Example:
        makeChunksFromInds([(0, 0), (0, 2), (2, 0), (2, 2)],(4,4),(2,2))
    Output:
        [[np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])],
         [np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])],
         [np.array([2, 2, 3, 3]), np.array([0, 1, 0, 1])],
         [np.array([2, 2, 3, 3]), np.array([2, 3, 2, 3])]]
    '''
    indices = []
    for ci in startInd:
        index = []
        for si, k, dim in zip(ci, chunkSize, range(len(chunkSize))):
            if si + k > in_shape[dim]:
                dend = in_shape[dim]
            else:
                dend = si + k
            index.append(np.array(range(si, dend)))
        indices.append(index)

    # Now create the index mesh (for Ndim > 1)
    chunks = []
    if len(in_shape) > 1:
        for index in indices:
            chunks.append([np.array(g) for g in zip(*list(itertools.product(*index)))])
    else:
        chunks = indices

    return chunks


def process_chunk(k, chunkInds, SP, SLV, chunkSize, stepSize, ifWet, ifHydro, zref, wm_file):
    """
    Perform the interpolation and integration over a single chunk.
    """
    # Transformer from ECEF to weather model
    p1 = CRS.from_epsg(4978)
    proj_wm = getProjFromWMFile(wm_file)
    t = Transformer.from_proj(p1, proj_wm, always_xy=True)

    # datatype must be specific for the cython makeRays* function
    _DTYPE = np.float64

    # H5PY does not support fancy indexing with tuples, hence this if/else check
    if len(chunkSize) == 1:
        row = chunkInds[0]
        ray = makeRays1D(zref, SP[row, :].astype(_DTYPE), SLV[row, :].astype(_DTYPE), stepSize)
    elif len(chunkSize) == 2:
        row, col = chunkInds
        ray = makeRays1D(zref, SP[row, col, :].astype(_DTYPE), SLV[row, col, :].astype(_DTYPE), stepSize)
    elif len(chunkSize) == 3:
        row, col, zind = chunkInds
        ray = makeRays1D(zref, SP[row, col, zind, :].astype(_DTYPE), SLV[row, col, zind, :].astype(_DTYPE), stepSize)
    else:
        raise RuntimeError('Data in more than 4 dimensions is not supported')

    ray_x, ray_y, ray_z = t.transform(ray[..., 0, :], ray[..., 1, :], ray[..., 2, :])
    delay_wet = interpolate2(ifWet, ray_x, ray_y, ray_z)
    delay_hydro = interpolate2(ifHydro, ray_x, ray_y, ray_z)
    int_delays = _integrateLOS(stepSize, delay_wet, delay_hydro)

    return int_delays


def getProjFromWMFile(wm_file):
    '''
    Returns the projection of an HDF5 file
    '''
    with h5py.File(wm_file, 'r') as f:
        wm_proj = CRS.from_json(f['Projection'][()])
    return wm_proj


def interpolate2(fun, x, y, z):
    '''
    helper function to make the interpolation step cleaner
    '''
    in_shape = x.shape
    out = fun((y.ravel(), x.ravel(), z.ravel()))  # note that this re-ordering is on purpose to match the weather model
    outData = out.reshape(in_shape)
    return outData


def _integrateLOS(stepSize, wet_pw, hydro_pw, Npts=None):
    delays = []
    for d in (wet_pw, hydro_pw):
        if d.ndim == 1:
            delays.append(np.array([int_fcn(d, stepSize)]))
        else:
            delays.append(_integrate_delays(stepSize, d, Npts))
    return np.stack(delays, axis=0)


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
            delays.append(int_fcn(ray, stepSize))
    return np.array(delays)


def int_fcn(y, dx, N=None):
    return 1e-6 * dx * np.nansum(y[:N])
