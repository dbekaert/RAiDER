#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson, David Bekaert & Yang Lei
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import logging
import multiprocessing as mp
import time

import h5py
import numpy as np
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

from RAiDER.constants import _STEP
from RAiDER.makePoints import makePoints1D

log = logging.getLogger(__name__)


def calculate_rays(pnts_file, stepSize=_STEP):
    '''
    From a set of lats/lons/hgts, compute ray paths from the ground to the
    top of the atmosphere, using either a set of look vectors or the zenith
    '''
    log.debug('calculate_rays: Starting look vector calculation')
    log.debug('The integration stepsize is %f m', stepSize)

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
    reproject a set of lat/lon/hgts to a new coordinate system
    '''

    t = Transformer.from_crs(4326, 4978, always_xy=True)  # converts from WGS84 geodetic to WGS84 geocentric
    with h5py.File(pnts_file, 'r+') as f:
        sp = np.moveaxis(np.array(t.transform(f['lon'][()], f['lat'][()], f['hgt'][()])), 0, -1)
        f['Rays_SP'][...] = sp.astype(np.float64)  # ensure double is maintained


def get_delays(stepSize, pnts_file, wm_file, interpType='3D',
               delayType="Zenith", cpu_num=0):
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

    ifWet = make_interpolator(xs_wm, ys_wm, zs_wm, wet)
    ifHydro = make_interpolator(xs_wm, ys_wm, zs_wm, hydro)

    with h5py.File(pnts_file, 'r') as f:
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
            individual_results = pool.starmap(process_chunk, chunk_inputs)

        delays = np.concatenate(individual_results)

    wet_delay = np.concatenate([d[0, ...] for d in delays]).reshape(in_shape)
    hydro_delay = np.concatenate([d[1, ...] for d in delays]).reshape(in_shape)

    time_elapse = (time.time() - t0)
    with open('get_delays_time_elapse.txt', 'w') as f:
        f.write('{}'.format(time_elapse))

    time_elapse_hr = int(np.floor(time_elapse/3600.0))
    time_elapse_min = int(np.floor((time_elapse - time_elapse_hr*3600.0)/60.0))
    time_elapse_sec = (time_elapse - time_elapse_hr*3600.0 - time_elapse_min*60.0)
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
        (ys.flatten(), xs.flatten(), zs.flatten()),
        data,
        bounds_error=False,
        fill_value=np.nan
    )


def process_chunk(k, chunkInds, SP, SLV, in_shape, stepSize, ifWet, ifHydro, max_len, wm_file):
    """
    Perform the interpolation and integration over a single chunk.
    """
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
    return 1e-6*dx*np.nansum(y[:N])
