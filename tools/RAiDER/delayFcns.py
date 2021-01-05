#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson, David Bekaert & Yang Lei
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import itertools
import multiprocessing as mp
import time

import h5py
from netCDF4 import Dataset
import numpy as np
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

from RAiDER.constants import _STEP
from RAiDER.interpolator import RegularGridInterpolator as Interpolator
from RAiDER.makePoints import makePoints1D


def calculate_rays(pnts_file, stepSize=_STEP):
    '''
    From a set of lats/lons/hgts, compute ray paths from the ground to the
    top of the atmosphere, using either a set of look vectors or the zenith
    '''
    # get the lengths of each ray for doing the interpolation
    getUnitLVs(pnts_file)

    # This projects the ground pixels into earth-centered, earth-fixed coordinate
    # system and sorts by position
    lla2ecef(pnts_file)


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


def lla2ecef(pnts_file):
    '''
    reproject a set of lat/lon/hgts to earth-centered, earth-fixed coordinate system
    '''
    t = Transformer.from_crs(4326, 4978, always_xy=True)  # converts from WGS84 geodetic to WGS84 geocentric
    
    with h5py.File(pnts_file, 'r+') as f:
        ndv = f.attrs['NoDataValue']
        lon = f['lon'][()]
        lat = f['lat'][()]
        hgt = f['hgt'][()]
        lon[lon == ndv] = np.nan
        lat[lat == ndv] = np.nan
        hgt[hgt == ndv] = np.nan
        sp = np.moveaxis(np.array(t.transform(lon, lat, hgt)), 0, -1)
        f['Rays_SP'][...] = sp.astype(np.float64)  # ensure double is maintained


def get_delays(
        stepSize, 
        pnts_file, 
        wm_file, 
        cpu_num=0
    ):
    '''
    Create the integration points for each ray path.
    '''

    t0 = time.time()

    # Get the weather model data
    with Dataset(wm_file, mode='r') as f:
        xs_wm = np.array(f.variables['x'][:])
        ys_wm = np.array(f.variables['y'][:])
        zs_wm = np.array(f.variables['z'][:])
        wet = np.array(f.variables['wet'][:]).swapaxes(1,2).swapaxes(0,2)
        hydro = np.array(f.variables['hydro'][:]).swapaxes(1,2).swapaxes(0,2)

    ifWet = Interpolator((ys_wm, xs_wm, zs_wm), wet, fill_value=np.nan)
    ifHydro = Interpolator((ys_wm, xs_wm, zs_wm), hydro, fill_value=np.nan)

    with h5py.File(pnts_file, 'r') as f:
        Nrays = f.attrs['NumRays']
        chunkSize = f.attrs['ChunkSize']
        in_shape = f['lon'].attrs['Shape']
        arrSize = f['lon'].shape
        max_len = np.nanmax(f['Rays_len'])

    CHUNKS = chunk(chunkSize, in_shape)
    Nchunks = len(CHUNKS)

    with h5py.File(pnts_file, 'r') as f:
        chunk_inputs = [(kk, CHUNKS[kk], np.array(f['Rays_SP']), np.array(f['Rays_SLV']),
                         chunkSize, stepSize, ifWet, ifHydro, max_len, wm_file) for kk in range(Nchunks)]

    if Nchunks == 1:
        delays = process_chunk(*chunk_inputs[0])
    else:
        with mp.Pool() as pool:
            individual_results = pool.starmap(process_chunk, chunk_inputs)
        try:
            delays = np.concatenate(individual_results)
        except ValueError:
            delays = np.concatenate(individual_results, axis=-1)

    wet_delay = delays[0, ...].reshape(in_shape)
    hydro_delay = delays[1, ...].reshape(in_shape)

    time_elapse = (time.time() - t0)
    with open('get_delays_time_elapse.txt', 'w') as f:
        f.write('{}'.format(time_elapse))
    time_elapse_hr = int(np.floor(time_elapse / 3600.0))
    time_elapse_min = int(np.floor((time_elapse - time_elapse_hr * 3600.0) / 60.0))
    time_elapse_sec = (time_elapse - time_elapse_hr * 3600.0 - time_elapse_min * 60.0)
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


def process_chunk(k, chunkInds, SP, SLV, chunkSize, stepSize, ifWet, ifHydro, max_len, wm_file):
    """
    Perform the interpolation and integration over a single chunk.
    """
    # Transformer from ECEF to weather model
    p1 = CRS.from_epsg(4978)
    proj_wm = getProjFromWMFile(wm_file)
    t = Transformer.from_proj(p1, proj_wm, always_xy=True)

    # datatype must be specific for the cython makePoints* function
    _DTYPE = np.float64

    # H5PY does not support fancy indexing with tuples, hence this if/else check
    if len(chunkSize) == 1:
        row = chunkInds[0]
        ray = makePoints1D(max_len, SP[row, :].astype(_DTYPE), SLV[row, :].astype(_DTYPE), stepSize)
    elif len(chunkSize) == 2:
        row, col = chunkInds
        ray = makePoints1D(max_len, SP[row, col, :].astype(_DTYPE), SLV[row, col, :].astype(_DTYPE), stepSize)
    elif len(chunkSize) == 3:
        row, col, zind = chunkInds
        ray = makePoints1D(max_len, SP[row, col, zind, :].astype(_DTYPE), SLV[row, col, zind, :].astype(_DTYPE), stepSize)
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
    with Dataset(wm_file, mode='r') as f:
        wm_proj = CRS.from_string(f.variables['WGS84'].spatial_ref)
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
