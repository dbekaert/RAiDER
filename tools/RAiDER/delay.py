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

from RAiDER.delayFcns import getInterpolators
from RAiDER.logger import logger
from RAiDER.losreader import getTopOfAtmosphere
from RAiDER.utilFcns import (
    lla2ecef,  writeResultsToXarray,
    rio_profile, transformPoints,
)

from RAiDER.constants import _ZREF


###############################################################################
def tropo_delay(
        dt,
        weather_model_file: str,
        aoi,
        los,
        height_levels: List[float]=None,
        out_proj: Union[int, str] =4326,
        zref: Union[int, float]=_ZREF,
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
        zref: int,float             - (optional) maximum height to integrate up to during raytracing

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
    ds = _get_delays_on_cube(dt, weather_model_file, wm_proj, aoi, height_levels,
            los, crs, zref)

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


def _get_delays_on_cube(dt, weather_model_file, wm_proj, aoi, heights, los, crs, zref, nproc=1):
    """
    raider cube generation function.
    """
    zpts = np.array(heights)

    # If no orbit is provided
    if los.is_Zenith() or los.is_Projected():
        out_type = ["zenith" if los.is_Zenith() else 'slant - projected'][0]

        # Get ZTD interpolators
        try:
            ifWet, ifHydro = getInterpolators(weather_model_file, "total")
        except RuntimeError:
            logger.exception('Weather model {} failed, may contain NaNs'.format(weather_model_file))


        # Build cube
        wetDelay, hydroDelay = _build_cube(
            aoi.xpts, aoi.ypts, zpts,
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
                aoi.xpts, aoi.ypts, zpts, los,
                wm_proj, crs, [ifWet, ifHydro], MAX_TROPO_HEIGHT=zref)


        ### Use multi-processing here
        else:
            # Pre-build output arrays

            # Create worker pool

            # Loop over heights
            raise NotImplementedError

    if np.isnan(wetDelay).any() or np.isnan(hydroDelay).any():
        raise Exception('There are missing delay values. Check your inputs. Not writing to disk.')

    # Write output file
    ds = writeResultsToXarray(dt, aoi.xpts, aoi.ypts, zpts, crs, wetDelay,
                              hydroDelay, weather_model_file, out_type)

    return ds


def _build_cube(xpts, ypts, zpts, model_crs, pts_crs, interpolators):
    """
    Iterate over interpolators and build a cube using Zenith
    """
    # Create a regular 2D grid
    xx, yy = np.meshgrid(xpts, ypts)

    # Output arrays
    outputArrs = [np.zeros((zpts.size, ypts.size, xpts.size))
                  for mm in range(len(interpolators))]


    # Loop over heights and compute delays
    for ii, ht in enumerate(zpts):

        # pts is in weather model system;
        if model_crs != pts_crs:
            # lat / lon / height for hrrr
            pts = transformPoints(yy, xx, np.full(yy.shape, ht),
                                               pts_crs, model_crs)
        else:
            pts = np.stack([yy, xx, np.full(yy.shape, ht)], axis=-1)

        for mm, intp in enumerate(interpolators):
            outputArrs[mm][ii,...] = intp(pts)

    return outputArrs


def _build_cube_ray(
        xpts, ypts, zpts, los, model_crs,
        pts_crs, interpolators, outputArrs=None, MAX_SEGMENT_LENGTH = 1000.,
        MAX_TROPO_HEIGHT = _ZREF,
    ):
    """
    Iterate over interpolators and build a cube using raytracing
    """
    # Get model heights in an array
    # Assumption: All interpolators here are on the same grid
    model_zs = interpolators[0].grid[2]

    # Create a regular 2D grid
    xx, yy = np.meshgrid(xpts, ypts)

    # Output arrays (1 for wet, 1 for hydro)
    output_created_here = False
    if outputArrs is None:
        output_created_here = True
        outputArrs = [np.zeros((zpts.size, ypts.size, xpts.size))
                      for mm in range(len(interpolators))]

    # Various transformers needed here
    epsg4326 = CRS.from_epsg(4326)
    cube_to_llh = Transformer.from_crs(pts_crs, epsg4326,
                                       always_xy=True)
    ecef_to_model = Transformer.from_crs(CRS.from_epsg(4978), model_crs,
                                         always_xy=True)

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

            # If low_ht < requested height, start integral at requested height
            if low_ht < ht:
                low_ht = ht

            # If high_ht > max_tropo_height - integral only up to max tropo
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

            # Integrate over chunks of ray
            for findex, ff in enumerate(fracs):
                # Ray point in ECEF coordinates
                pts_xyz = low_xyz + ff * (high_xyz - low_xyz)

                # Ray point in model coordinates (x, y, z)
                pts = ecef_to_model.transform(
                    pts_xyz[..., 0],
                    pts_xyz[..., 1],
                    pts_xyz[..., 2]
                )

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
                wt = 0.5 if findex in [0, fracs.size-1] else 1.0
                wt *= ray_length *1.0e-6 / (nParts - 1.0)

                # For each interpolator, integrate between levels
                for mm, out in enumerate(outSubs):
                    val  =  interpolators[mm](pts)
                    out += wt * val

    if output_created_here:
        return outputArrs
