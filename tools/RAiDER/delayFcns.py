# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson, David Bekaert & Yang Lei
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import itertools
import multiprocessing as mp
import xarray

from netCDF4 import Dataset
import numpy as np
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator as Interpolator

from RAiDER.constants import _STEP
from RAiDER.makePoints import makePoints1D


def calculate_start_points(x, y, z, ds):
    '''
    Args:
    ----------
    wm_file: str   - A file containing a regularized weather model.
    Returns:
    -------
    SP: ndarray    - a * x 3 array containing the XYZ locations of the pixels in ECEF coordinates.
                     Note the ordering of the array is [Y X Z]
    '''
    [X, Y, Z] = np.meshgrid(x, y, z)

    try:
        t = Transformer.from_crs(ds['CRS'], 4978, always_xy=True)  # converts to WGS84 geocentric
    except:
        print("I can't find a CRS in the weather model file, so I will assume you are using WGS84")
        t = Transformer.from_crs(4326, 4978, always_xy=True)  # converts to WGS84 geocentric

    return np.moveaxis(np.array(t.transform(X, Y, Z)), 0, -1), np.stack([X, Y, Z], axis=-1)


def get_delays(
    stepSize,
    SP,
    LOS,
    wm_file,
    cpu_num=0
):
    '''
    Create the integration points for each ray path.
    '''
    ifWet, ifHydro = getInterpolators(wm_file)

    with xarray.load_dataset(wm_file) as f:
        try:
            wm_proj = f.attrs['CRS']
        except:
            wm_proj = 4326
            print("I can't find a CRS in the weather model file, so I will assume you are using WGS84")
            t = Transformer.from_crs(4326, 4978, always_xy=True)  # converts to WGS84 geocentric

    in_shape = SP.shape[:-1]
    chunkSize = in_shape
    CHUNKS = chunk(in_shape, in_shape)
    Nchunks = len(CHUNKS)
    max_len = 15000
    stepSize = 100

    chunk_inputs = [(kk, CHUNKS[kk], wm_proj, SP, LOS,
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

    return wet_delay, hydro_delay


def getInterpolators(wm_file, kind='pointwise', shared=False):
    '''
    Read 3D gridded data from a processed weather model file and wrap it with
    an interpolator
    '''
    # Get the weather model data
    try:
        ds = xarray.load_dataset(wm_file)
    except:
        ds = wm_file

    xs_wm = np.array(ds.variables['x'][:])
    ys_wm = np.array(ds.variables['y'][:])
    zs_wm = np.array(ds.variables['z'][:])
    wet = ds.variables['wet_total' if kind=='total' else 'wet'][:]
    hydro = ds.variables['hydro_total' if kind=='total' else 'hydro'][:]

    wet = np.array(wet).transpose(1, 2, 0)
    hydro = np.array(hydro).transpose(1, 2, 0)

    # If shared interpolators are requested
    # The arrays are not modified - so turning off lock for performance
    if shared:
        xs_wm = make_shared_raw(xs_wm)
        ys_wm = make_shared_raw(ys_wm)
        zs_wm = make_shared_raw(zs_wm)
        wet = make_shared_raw(wet)
        hydro = make_shared_raw(hydro)

    ifWet = Interpolator((ys_wm, xs_wm, zs_wm), wet, fill_value=np.nan, bounds_error = False)
    ifHydro = Interpolator((ys_wm, xs_wm, zs_wm), hydro, fill_value=np.nan, bounds_error = False)

    return ifWet, ifHydro


def make_shared_raw(inarr):
    """
    Make numpy view array of mp.Array
    """
    # Create flat shared array
    shared_arr = mp.RawArray('d', inarr.size)
    # Create a numpy view of it
    shared_arr_np = np.ndarray(inarr.shape, dtype=np.float64,
                               buffer=shared_arr)
    # Copy data to shared array
    np.copyto(shared_arr_np, inarr)

    return shared_arr_np


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


def process_chunk(k, chunkInds, proj_wm, SP, SLV, chunkSize, stepSize, ifWet, ifHydro, max_len, wm_file):
    """
    Perform the interpolation and integration over a single chunk.
    """
    # Transformer from ECEF to weather model
    t = Transformer.from_proj(4978, proj_wm, always_xy=True)

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
