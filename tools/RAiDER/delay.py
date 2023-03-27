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
import pyproj
import xarray

from pyproj import CRS, Transformer
from typing import List, Union

import numpy as np

from RAiDER.delayFcns import (
    getInterpolators, get_output_spacing,
)
from RAiDER.logger import logger
from RAiDER.losreader import getTopOfAtmosphere
from RAiDER.utilFcns import (
    lla2ecef, transform_bbox, clip_bbox, writeResultsToXarray,
    rio_profile,
)


###############################################################################
def tropo_delay(
        dt,
        weather_model_file: str,
        aoi,
        los,
        height_levels: List[float]=None,
        out_proj: Union[int, str] =4326,
        cube_spacing_m: int=None,
        look_dir: str='right',
    ):
    """
    Calculate integrated delays on query points. Options are:
    1. Zenith delays (ZTD)
    2. Zenith delays projected to the line-of-sight (STD-projected)
    3. Slant delays integrated along the raypath (STD-raytracing)

    Args:
        dt: Datetime                - Datetime object for determining when to calculate delays
        weather_model_File: string  - Name of the NETCDF file containing a pre-processed weather model
        aoi: AOI object             - AOI object
        los: LOS object             - LOS object
        height_levels: list         - (optional) list of height levels on which to calculate delays. Only needed for cube generation.
        out_proj: int,str           - (optional) EPSG code for output projection
        look_dir: str               - (optional) Satellite look direction. Only needed for slant delay calculation
        cube_spacing_m: int         - (optional) Horizontal spacing in meters when generating cubes

    Returns:
        xarray Dataset *or* ndarrays: - wet and hydrostatic delays at the grid nodes / query points.
    """
    crs = CRS(out_proj)

    # Load CRS from weather model file
    wm_proj = rio_profile(f"netcdf:{weather_model_file}:t")["crs"]
    if wm_proj is None:
       logger.warning("WARNING: I can't find a CRS in the weather model file, so I will assume you are using WGS84")
       wm_proj = CRS.from_epsg(4326)
    else:
        wm_proj = CRS.from_wkt(wm_proj.to_wkt())

    # get heights
    if height_levels is None:
        if aoi.type() == 'Geocube':
            height_levels = aoi.readZ()

        else:
            with xarray.load_dataset(weather_model_file) as ds:
                height_levels = ds.z.values

    #TODO: expose this as library function
    ds = _get_delays_on_cube(dt, weather_model_file, wm_proj, aoi.bounds(), height_levels,
            los, crs=crs, cube_spacing_m=cube_spacing_m, look_dir=look_dir)

    if (aoi.type() == 'bounding_box') or (aoi.type() == 'Geocube'):
        return ds, None

    else:
        # CRS can be an int, str, or CRS object
        try:
            out_proj = CRS.from_epsg(out_proj)
        except pyproj.exceptions.CRSError:
            out_proj = out_proj

        pnt_proj = CRS.from_epsg(4326)
        lats, lons = aoi.readLL()
        hgts = aoi.readZ()
        pnts = transformPoints(lats, lons, hgts, pnt_proj, out_proj)
        if pnts.ndim == 3:
            pnts = pnts.transpose(2,1,0)
        elif pnts.ndim == 2:
            pnts = pnts.T
        try:
            ifWet, ifHydro = getInterpolators(ds, "ztd")
        except RuntimeError:
            logger.exception('Weather model %s failed, may contain NaNs', weather_model_file)
        wetDelay = ifWet(pnts)
        hydroDelay = ifHydro(pnts)

        # return the delays (ZTD or STD)
        if los.is_Projected():
            los.setTime(dt)
            los.setPoints(lats, lons, hgts)
            wetDelay   = los(wetDelay)
            hydroDelay = los(hydroDelay)

    return wetDelay, hydroDelay


def _get_delays_on_cube(dt, weather_model_file, wm_proj, ll_bounds, heights, los, crs, cube_spacing_m=None, look_dir='right', nproc=1):
    """
    raider cube generation function.
    """

    # Determine the output grid extent here and clip output grid to multiples of spacing

    snwe = transform_bbox(ll_bounds, src_crs=4326, dest_crs=crs)
    out_spacing = get_output_spacing(cube_spacing_m, weather_model_file, wm_proj, crs)
    out_snwe = clip_bbox(snwe, out_spacing)

    logger.debug(f"Output SNWE: {out_snwe}")
    logger.debug(f"Output cube spacing: {out_spacing}")

    # Build the output grid
    zpts = np.array(heights)
    xpts = np.arange(out_snwe[2], out_snwe[3] + out_spacing, out_spacing)
    ypts = np.arange(out_snwe[1], out_snwe[0] - out_spacing, -out_spacing)

    # If no orbit is provided
    # Build zenith delay cube
    if los.is_Zenith() or los.is_Projected():
        out_type = ["zenith" if los.is_Zenith() else 'slant - projected'][0]

        # Get ZTD interpolators
        try:
            ifWet, ifHydro = getInterpolators(weather_model_file, "total")
        except RuntimeError:
            logger.exception('Weather model {} failed, may contain NaNs'.format(weather_model_file))


        # Build cube
        wetDelay, hydroDelay = _build_cube(
            xpts, ypts, zpts,
            wm_proj, crs,
            [ifWet, ifHydro])

    else:
        out_type = "slant - raytracing"

        # Get pointwise interpolators
        try:
            ifWet, ifHydro = getInterpolators(
                weather_model_file,
                kind="pointwise",
                shared=(nproc > 1),
            )
        except RuntimeError:
            logger.exception('Weather model {} failed, may contain NaNs'.format(weather_model_file))

        # Build cube
        if nproc == 1:
            wetDelay, hydroDelay = _build_cube_ray(
                xpts, ypts, zpts, los,
                wm_proj, crs,
                [ifWet, ifHydro])

        ### Use multi-processing here
        else:
            # Pre-build output arrays

            # Create worker pool

            # Loop over heights
            raise NotImplementedError

    # Write output file
    ds = writeResultsToXarray(dt, xpts, ypts, zpts, crs, wetDelay, hydroDelay, weather_model_file, out_type)

    return ds


def transformPoints(lats: np.ndarray, lons: np.ndarray, hgts: np.ndarray, old_proj: CRS, new_proj: CRS) -> np.ndarray:
    '''
    Transform lat/lon/hgt data to an array of points in a new
    projection

    Args:
        lats: ndarray   - WGS-84 latitude (EPSG: 4326)
        lons: ndarray   - ditto for longitude
        hgts: ndarray   - Ellipsoidal height in meters
        old_proj: CRS   - the original projection of the points
        new_proj: CRS   - the new projection in which to return the points

    Returns:
        ndarray: the array of query points in the weather model coordinate system (YX)
    '''
    t = Transformer.from_crs(old_proj, new_proj)

    # Flags for flipping inputs or outputs
    if not isinstance(new_proj, pyproj.CRS):
        new_proj = CRS.from_epsg(new_proj.lstrip('EPSG:'))
    if not isinstance(old_proj, pyproj.CRS):
        old_proj = CRS.from_epsg(old_proj.lstrip('EPSG:'))

    in_flip = old_proj.axis_info[0].direction
    out_flip = new_proj.axis_info[0].direction

    if in_flip == 'east':
        res = t.transform(lons, lats, hgts)
    else:
        res = t.transform(lats, lons, hgts)

    if out_flip == 'east':
        return np.stack((res[1], res[0], res[2]), axis=-1).T
    else:
        return np.stack(res, axis=-1).T


def _build_cube(xpts, ypts, zpts, model_crs, pts_crs, interpolators):
    """
    Iterate over interpolators and build a cube using Zenith
    """
    # Create a regular 2D grid
    xx, yy = np.meshgrid(xpts, ypts)

    # Output arrays
    outputArrs = [np.zeros((zpts.size, ypts.size, xpts.size))
                  for mm in range(len(interpolators))]

    # Assume all interpolators to be on same grid
    zmin = min(interpolators[0].grid[2])
    zmax = max(interpolators[0].grid[2])

    # print("Output grid: ")
    # print("crs: ", pts_crs)
    # print("X: ", xpts[0], xpts[-1])
    # print("Y: ", ypts[0], ypts[-1])

    # ii = interpolators[0]
    # print("Model grid: ")
    # print("crs: ", model_crs)
    # print("X: ", ii.grid[1][0], ii.grid[1][-1])
    # print("Y: ", ii.grid[0][0], ii.grid[0][-1])

    # Loop over heights and compute delays
    for ii, ht in enumerate(zpts):

        # Uncomment this line to clip heights for interpolator
        # ht = max(zmin, min(ht, zmax))

        # pts is in weather model system
        if model_crs != pts_crs:
            pts = np.transpose(
                transformPoints(
                    yy, xx, np.full(yy.shape, ht),
                    pts_crs, model_crs,
                ), (2, 1, 0))
        else:
            pts = np.stack([yy, xx, np.full(yy.shape, ht)], axis=-1)

        for mm, intp in enumerate(interpolators):
            outputArrs[mm][ii,...] = intp(pts)

    return outputArrs


def _build_cube_ray(
        xpts, ypts, zpts, los, model_crs,
        pts_crs, interpolators, outputArrs=None, MAX_SEGMENT_LENGTH = 1000.,
        MAX_TROPO_HEIGHT = 50000.,
    ):
    """
    Iterate over interpolators and build a cube using raytracing
    """
    # Get model heights in an array
    # Assumption: All interpolators here are on the same grid
    model_zs = interpolators[0].grid[2]

    # Create a regular 2D grid
    xx, yy = np.meshgrid(xpts, ypts)

    # Output arrays
    output_created_here = False
    if outputArrs is None:
        output_created_here = True
        outputArrs = [np.zeros((zpts.size, ypts.size, xpts.size))
                      for mm in range(len(interpolators))]

    # Various transformers needed here
    epsg4326 = CRS.from_epsg(4326)
    cube_to_llh = Transformer.from_crs(pts_crs, epsg4326,
                                       always_xy=True)
    ecef_to_model = Transformer.from_crs(CRS.from_epsg(4978), model_crs)
    # For calling the interpolators
    flip_xy = model_crs.axis_info[0].direction == "east"

    # Loop over heights of output cube and compute delays
    for hh, ht in enumerate(zpts):
        logger.info(f"Processing slice {hh+1} / {len(zpts)}: {ht}")
        # Slices to fill on output
        outSubs = [x[hh, ...] for x in outputArrs]

        # Step 1:  transform points to llh and xyz
        if pts_crs != epsg4326:
            llh = list(cube_to_llh.transform(xx, yy, np.full(yy.shape, ht)))
        else:
            llh = [xx, yy, np.full(yy.shape, ht)]

        xyz = np.stack(lla2ecef(llh[1], llh[0], np.full(yy.shape, ht)), axis=-1)

        # Step 2 - get LOS vectors for targets
        LOS = los.getLookVectors(ht, llh, xyz, yy)

        # Step 3 - Determine delays between each model height per ray
        # Assumption: zpts (output cube) and model_zs (model) are assumed to be
        # sorted in height
        # We start integrating bottom up
        low_xyz = None
        high_xyz = None
        cos_factor = None
        for zz in range(model_zs.size-1):
            # Low and High for model interval
            low_ht = model_zs[zz]
            high_ht = model_zs[zz + 1]

            # If high_ht < height of point - no contribution to integral
            # If low_ht > max_tropo_height - no contribution to integral
            if (high_ht <= ht) or (low_ht >= MAX_TROPO_HEIGHT):
                continue

            # If low_ht < height of point - integral only up to height of point
            if low_ht < ht:
                low_ht = ht

            # If high_ht > max_tropo_height - integral only up to max tropo
            # height
            if high_ht > MAX_TROPO_HEIGHT:
                high_ht = MAX_TROPO_HEIGHT

            # Continue only if needed - 1m troposphere does nothing
            if np.abs(high_ht - low_ht) < 1.0:
                continue

            # If high_xyz was defined, make new low_xyz - save computation
            if high_xyz is not None:
                low_xyz = high_xyz
            else:
                low_xyz = getTopOfAtmosphere(xyz, LOS, low_ht, factor=cos_factor)

            # Compute high_xyz (upper model level)
            high_xyz = getTopOfAtmosphere(xyz, LOS, high_ht, factor=cos_factor)

            # Compute ray length
            ray_length =  np.linalg.norm(high_xyz - low_xyz, axis=-1)

            # Compute cos_factor for first iteration
            if cos_factor is None:
                cos_factor = (high_ht - low_ht) / ray_length

            # Determine number of parts to break ray into (this is what gets integrated over)
            try:
                nParts = int(np.ceil(ray_length.max() / MAX_SEGMENT_LENGTH)) + 1
            except ValueError:
                raise ValueError("geo2rdr did not converge. Check orbit coverage")

            if (nParts == 1):
                raise RuntimeError("Ray with one segment encountered")

            # fractions
            fracs = np.linspace(0., 1., num=nParts)

            # Integrate over the ray
            for findex, ff in enumerate(fracs):
                # Ray point in ECEF coordinates
                pts_xyz = low_xyz + ff * (high_xyz - low_xyz)

                # Ray point in model coordinates
                pts = ecef_to_model.transform(
                    pts_xyz[..., 0],
                    pts_xyz[..., 1],
                    pts_xyz[..., 2]
                )

                # Order for the interpolator
                if flip_xy:
                    pts = np.stack((pts[1], pts[0], pts[2]), axis=-1)
                else:
                    pts = np.stack(pts, axis=-1)

                # Trapezoidal integration with scaling
                wt = 0.5 if findex in [0, fracs.size-1] else 1.0
                wt *= ray_length *1.0e-6 / (nParts - 1.0)

                # For each interpolator, integrate between levels
                for mm, out in enumerate(outSubs):
                    val =  interpolators[mm](pts)

                    # TODO - This should not occur if there is enough padding in model
                    val[np.isnan(val)] = 0.0
                    out += wt * val

    if output_created_here:
        return outputArrs
