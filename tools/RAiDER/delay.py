# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson, David Bekaert & Yang Lei
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""RAiDER tropospheric delay calculation

This module provides the main RAiDER functionality for calculating
tropospheric wet and hydrostatic delays from a weather model. Weather
models are accessed as NETCDF files and should have "wet" "hydro"
"wet_total" and "hydro_total" fields specified.
"""

import datetime as dt
import os
from typing import Optional, Union

from RAiDER.types import CRSLike
from RAiDER.utilFcns import parse_crs
import numpy as np
import pyproj
import xarray as xr
from pyproj import CRS, Transformer

from RAiDER.constants import _ZREF
from RAiDER.delayFcns import getInterpolators
from RAiDER.llreader import AOI, BoundingBox, Geocube
from RAiDER.logger import logger
from RAiDER.losreader import LOS, build_ray
from RAiDER.types import CRSLike
from RAiDER.utilFcns import parse_crs


###############################################################################
def tropo_delay(
    datetime: dt.datetime,
    weather_model_file: str,
    aoi: AOI,
    los: LOS,
    height_levels: Optional[list[float]] = None,
    out_proj: Union[int, str] = 4326,
    zref: Optional[np.float64] = None,
):
    """Calculate integrated delays on query points.
    
    Options are:
    1. Zenith delays (ZTD)
    2. Zenith delays projected to the line-of-sight (STD-projected)
    3. Slant delays integrated along the raypath (STD-raytracing)

    Args:
        datetime: Datetime          - Datetime object for determining when to calculate delays
        weather_model_File: string  - Name of the NETCDF file containing a pre-processed weather model
        aoi: AOI object             - AOI object
        los: LOS object             - LOS object
        height_levels: list         - (optional) list of height levels on which to calculate delays. Only needed for cube generation.
        out_proj: int,str           - (optional) EPSG code for output projection
        zref: int,float             - (optional) maximum height to integrate up to during raytracing

    Returns:
        xarray Dataset *or* ndarrays: - wet and hydrostatic delays at the grid nodes / query points.
    """
    crs = CRS(out_proj)

    # Load CRS from weather model file
    with xr.load_dataset(weather_model_file) as ds:
        try:
            wm_proj = CRS.from_wkt(ds['proj'].attrs['crs_wkt'])
        except KeyError:
            logger.warning(
                "WARNING: I can't find a CRS in the weather model file, so I will assume you are using WGS84"
            )
            wm_proj = CRS.from_epsg(4326)

    # get heights
    with xr.load_dataset(weather_model_file) as ds:
        wm_levels = ds.z.values
        toa = wm_levels.max() - 1

    if height_levels is None:
        if isinstance(aoi, Geocube):
            height_levels = aoi.readZ()
        else:
            height_levels = wm_levels

    if zref is None:
        zref = toa

    if zref > toa:
        zref = toa
        logger.warning(
            'Requested integration height (zref) is higher than top of weather model. Forcing to top ({toa}).'
        )

    # TODO: expose this as library function
    ds = _get_delays_on_cube(datetime, weather_model_file, wm_proj, aoi, height_levels, los, crs, zref)

    if isinstance(aoi, (BoundingBox, Geocube)):
        return ds, None

    else:
        # CRS can be an int, str, or CRS object
        try:
            out_proj = CRS.from_epsg(out_proj)
        except pyproj.exceptions.CRSError:
            pass

        pnt_proj = CRS.from_epsg(4326)
        lats, lons = aoi.readLL()
        hgts = aoi.readZ()
        pnts = transformPoints(lats, lons, hgts, pnt_proj, out_proj)

        try:
            ifWet, ifHydro = getInterpolators(ds, 'ztd')
        except RuntimeError:
            raise RuntimeError(f'Failed to get weather model {weather_model_file} interpolators.')

        wetDelay = ifWet(pnts)
        hydroDelay = ifHydro(pnts)

        # return the delays (ZTD or STD)
        if los.is_Projected():
            los.setTime(datetime)
            los.setPoints(lats, lons, hgts)
            wetDelay = los(wetDelay)
            hydroDelay = los(hydroDelay)

    return wetDelay, hydroDelay


def _get_delays_on_cube(datetime: dt.datetime, weather_model_file, wm_proj, aoi, heights, los, crs, zref, nproc=1):
    """Raider cube generation function."""
    zpts = np.array(heights)

    try:
        aoi.xpts
    except AttributeError:
        with xr.load_dataset(weather_model_file) as ds:
            x_spacing = ds.x.diff(dim='x').values.mean()
            y_spacing = ds.y.diff(dim='y').values.mean()
        aoi.set_output_spacing(ll_res=np.min([x_spacing, y_spacing]))
        aoi.set_output_xygrid(crs)

    # If no orbit is provided
    if los.is_Zenith() or los.is_Projected():
        out_type = ['zenith' if los.is_Zenith() else 'slant - projected'][0]

        # Get ZTD interpolators
        try:
            ifWet, ifHydro = getInterpolators(weather_model_file, 'total')
        except RuntimeError:
            logger.exception('Failed to get weather model %s interpolators.', weather_model_file)

        # Build cube
        wetDelay, hydroDelay = _build_cube(aoi.xpts, aoi.ypts, zpts, wm_proj, crs, [ifWet, ifHydro])

    else:
        out_type = 'slant - raytracing'

        # Get pointwise interpolators
        try:
            ifWet, ifHydro = getInterpolators(
                weather_model_file,
                kind='pointwise',
                shared=(nproc > 1),
            )
        except RuntimeError:
            logger.exception('Failed to get weather model %s interpolators.', weather_model_file)

        # Build cube
        if nproc == 1:
            wetDelay, hydroDelay = _build_cube_ray(
                aoi.xpts, aoi.ypts, zpts, los, wm_proj, crs, [ifWet, ifHydro], MAX_TROPO_HEIGHT=zref
            )

        ### Use multi-processing here
        else:
            # Pre-build output arrays

            # Create worker pool

            # Loop over heights
            raise NotImplementedError

    if np.isnan(wetDelay).any() or np.isnan(hydroDelay).any():
        logger.critical('There are missing delay values. Check your inputs.')

    # Write output file
    ds = writeResultsToXarray(datetime, aoi.xpts, aoi.ypts, zpts, crs, wetDelay, hydroDelay, weather_model_file, out_type)

    return ds


def _build_cube(xpts, ypts, zpts, model_crs, pts_crs, interpolators):
    """Iterate over interpolators and build a cube using Zenith."""
    # Create a regular 2D grid
    xx, yy = np.meshgrid(xpts, ypts)

    # Output arrays
    outputArrs = [np.zeros((zpts.size, ypts.size, xpts.size)) for mm in range(len(interpolators))]

    # Loop over heights and compute delays
    for ii, ht in enumerate(zpts):
        # pts is in weather model system;
        if model_crs != pts_crs:
            # lat / lon / height for hrrr
            pts = transformPoints(yy, xx, np.full(yy.shape, ht), pts_crs, model_crs)
        else:
            pts = np.stack([yy, xx, np.full(yy.shape, ht)], axis=-1)

        for mm, intp in enumerate(interpolators):
            outputArrs[mm][ii, ...] = intp(pts)

    return outputArrs


def _build_cube_ray(
    xpts,
    ypts,
    zpts,
    los,
    model_crs,
    pts_crs,
    interpolators,
    outputArrs=None,
    MAX_SEGMENT_LENGTH=1000.0,
    MAX_TROPO_HEIGHT=_ZREF,
):
    """
    Iterate over interpolators and build a cube using raytracing.

    MAX_TROPO_HEIGHT should not extend above the top of the weather model
    """
    # Get model heights in an array
    # Assumption: All interpolators here are on the same grid
    T = Transformer.from_crs(4326, 4978, always_xy=True)
    model_zs = interpolators[0].grid[2]

    # Create a regular 2D grid
    xx, yy = np.meshgrid(xpts, ypts)

    # Output arrays (1 for wet, 1 for hydro)
    output_created_here = False
    if outputArrs is None:
        output_created_here = True
        outputArrs = [np.zeros((zpts.size, ypts.size, xpts.size)) for mm in range(len(interpolators))]

    # Various transformers needed here
    epsg4326 = CRS.from_epsg(4326)
    cube_to_llh = Transformer.from_crs(pts_crs, epsg4326, always_xy=True)
    ecef_to_model = Transformer.from_crs(CRS.from_epsg(4978), model_crs, always_xy=True)

    # Loop over heights of output cube and compute delays
    for hh, ht in enumerate(zpts):
        logger.info(f'Processing slice {hh+1} / {len(zpts)}: {ht}')
        # Slices to fill on output
        outSubs = [x[hh, ...] for x in outputArrs]

        # Step 1:  transform points to llh and xyz
        if pts_crs != epsg4326:
            llh = list(cube_to_llh.transform(xx, yy, np.full(yy.shape, ht)))
        else:
            llh = [xx, yy, np.full(yy.shape, ht)]

        xyz = np.stack(T.transform(llh[0], llh[1], llh[2]), axis=-1)

        # Step 2 - get LOS vectors for targets
        LOS = los.getLookVectors(ht, llh, xyz, yy)

        # Step 3 - Determine delays between each model height per ray
        ray_lengths, low_xyzs, high_xyzs = build_ray(model_zs, ht, xyz, LOS, MAX_TROPO_HEIGHT)

        # if the top most height layer doesnt contribute to the integral, skip it
        if ray_lengths is None and ht == zpts[-1]:
            continue

        elif np.isnan(ray_lengths).all():
            raise ValueError('geo2rdr did not converge. Check orbit coverage')

        # Determine number of parts to break ray into (this is what gets integrated over)
        nParts = np.ceil(ray_lengths.max((1, 2)) / MAX_SEGMENT_LENGTH).astype(int) + 1

        # iterate over weather model height levels
        for zz, nparts in enumerate(nParts):
            fracs = np.linspace(0.0, 1.0, num=nparts)

            # Integrate over chunks of ray
            for findex, ff in enumerate(fracs):
                # Ray point in ECEF coordinates
                pts_xyz = low_xyzs[zz] + ff * (high_xyzs[zz] - low_xyzs[zz])

                # Ray point in model coordinates (x, y, z)
                pts = ecef_to_model.transform(pts_xyz[..., 0], pts_xyz[..., 1], pts_xyz[..., 2])

                # Order for the interpolator (from xyz to yxz)
                pts = np.stack((pts[1], pts[0], pts[2]), axis=-1)

                # ray points first exist in ECEF; they are then projected to WGS84
                # this adds slight error (order 1 mm for 500 m)
                # at the lowest weather model layer (-500 m) the error pushes the
                # ray points slightly below (e.g., -500.0002 m)
                # the subsequent interpolation then results in nans
                # here we force the lowest layer up to -500 m if it exceeds it
                if (pts[:, :, -1] < np.array(model_zs).min()).all():
                    pts[:, :, -1] = np.array(model_zs).min()

                # same thing for upper bound
                if (pts[:, :, -1] > np.array(model_zs).max()).all():
                    pts[:, :, -1] = np.array(model_zs).max()

                # Trapezoidal integration with scaling
                wt = 0.5 if findex in [0, fracs.size - 1] else 1.0
                wt *= ray_lengths[zz] * 1.0e-6 / (nparts - 1.0)

                # For each interpolator, integrate between levels
                for mm, out in enumerate(outSubs):
                    val = interpolators[mm](pts)

                    # TODO - This should not occur if there is enough padding in model
                    # val[np.isnan(val)] = 0.0
                    out += wt * val

    if output_created_here:
        return outputArrs


def writeResultsToXarray(datetime: dt.datetime, xpts, ypts, zpts, crs, wetDelay, hydroDelay, weather_model_file, out_type):
    """Write a 1-D array to a NETCDF5 file."""
    # Modify this as needed for NISAR / other projects
    ds = xr.Dataset(
        data_vars=dict(
            wet=(
                ['z', 'y', 'x'],
                wetDelay,
                {
                    'units': 'm',
                    'description': f'wet {out_type} delay',
                    # 'crs': crs.to_epsg(),
                    'grid_mapping': 'crs',
                },
            ),
            hydro=(
                ['z', 'y', 'x'],
                hydroDelay,
                {
                    'units': 'm',
                    # 'crs': crs.to_epsg(),
                    'description': f'hydrostatic {out_type} delay',
                    'grid_mapping': 'crs',
                },
            ),
        ),
        coords=dict(
            x=(['x'], xpts),
            y=(['y'], ypts),
            z=(['z'], zpts),
        ),
        attrs=dict(
            Conventions='CF-1.7',
            title='RAiDER geo cube',
            source=os.path.basename(weather_model_file),
            history=str(dt.datetime.now(tz=dt.timezone.utc)) + ' RAiDER',
            description=f'RAiDER geo cube - {out_type}',
            reference_time=datetime.strftime('%Y%m%dT%H:%M:%S'),
        ),
    )

    # Write projection system mapping
    ds['crs'] = -2147483647  # dummy placeholder
    for k, v in crs.to_cf().items():
        ds.crs.attrs[k] = v

    # Write z-axis information
    ds.z.attrs['axis'] = 'Z'
    ds.z.attrs['units'] = 'm'
    ds.z.attrs['description'] = 'height above ellipsoid'

    # If in degrees
    if crs.axis_info[0].unit_name == 'degree':
        ds.y.attrs['units'] = 'degrees_north'
        ds.y.attrs['standard_name'] = 'latitude'
        ds.y.attrs['long_name'] = 'latitude'

        ds.x.attrs['units'] = 'degrees_east'
        ds.x.attrs['standard_name'] = 'longitude'
        ds.x.attrs['long_name'] = 'longitude'

    else:
        ds.y.attrs['axis'] = 'Y'
        ds.y.attrs['standard_name'] = 'projection_y_coordinate'
        ds.y.attrs['long_name'] = 'y-coordinate in projected coordinate system'
        ds.y.attrs['units'] = 'm'

        ds.x.attrs['axis'] = 'X'
        ds.x.attrs['standard_name'] = 'projection_x_coordinate'
        ds.x.attrs['long_name'] = 'x-coordinate in projected coordinate system'
        ds.x.attrs['units'] = 'm'

    return ds


def transformPoints(
    lats: np.ndarray,
    lons: np.ndarray,
    hgts: np.ndarray,
    old_proj: CRSLike,
    new_proj: CRSLike,
) -> np.ndarray:
    """
    Transform lat/lon/hgt data to an array of points in a new projection.

    Args:
        lats: ndarray   - WGS-84 latitude (EPSG: 4326)
        lons: ndarray   - ditto for longitude
        hgts: ndarray   - Ellipsoidal height in meters
        old_proj: CRS   - the original projection of the points
        new_proj: CRS   - the new projection in which to return the points

    Returns:
        ndarray: the array of query points in the weather model coordinate system (YX)
    """
    # Flags for flipping inputs or outputs
    old_proj = parse_crs(old_proj)
    new_proj = parse_crs(new_proj)

    t = Transformer.from_crs(old_proj, new_proj, always_xy=True)

    # in_flip = old_proj.axis_info[0].direction
    # out_flip = new_proj.axis_info[0].direction

    res = t.transform(lons, lats, hgts)

    # lat/lon/height
    return np.stack([res[1], res[0], res[2]], axis=-1)
