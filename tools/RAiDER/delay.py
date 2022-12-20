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
import os

import datetime
import pyproj
import xarray

from pyproj import CRS, Transformer
from typing import List

import isce3.ext.isce3 as isce
import numpy as np

from RAiDER.delayFcns import (
    getInterpolators
)
from RAiDER.logger import logger, logging
from RAiDER.losreader import get_sv, getTopOfAtmosphere
from RAiDER.utilFcns import (
    lla2ecef, transform_bbox, clip_bbox, rio_profile,
)


###############################################################################
def tropo_delay(
        dt, 
        weather_model_file: str, 
        aoi, 
        los, 
        height_levels: List[float]=None, 
        out_proj: int | str=4326, 
        cube_spacing_m: int=None,
        look_dir: str='right', 
    ):
    """
    Calculate integrated delays on query points.

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
    # get heights
    if height_levels is None:
        with xarray.load_dataset(weather_model_file) as ds:
            height_levels = ds.z.values

    #TODO: expose this as library function
    ds = _get_delays_on_cube(dt, weather_model_file, aoi.bounds(), height_levels,
            los, out_proj=out_proj, cube_spacing_m=cube_spacing_m, look_dir=look_dir)

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
            pnts = pnts.transpose(1,2,0)
        elif pnts.ndim == 2:
            pnts = pnts.T
        ifWet, ifHydro = getInterpolators(ds, 'ztd') # the cube from get_delays_on_cube calls the total delays 'wet' and 'hydro'
        wetDelay = ifWet(pnts)
        hydroDelay = ifHydro(pnts)

        # return the delays (ZTD or STD)
        if los.is_Projected():
            los.setTime(dt)
            los.setPoints(lats, lons, hgts)
            wetDelay   = los(wetDelay)
            hydroDelay = los(hydroDelay)

    return wetDelay, hydroDelay


def _get_delays_on_cube(dt, weather_model_file, ll_bounds, heights, los, out_proj=4326, cube_spacing_m=None, look_dir='right', nproc=1):
    """
    raider cube generation function.
    """
    # For testing multiprocessing
    # TODO - move this to configuration
    crs = CRS(out_proj)

    # Load CRS from weather model file
    wm_proj = rio_profile(f"netcdf:{weather_model_file}:t")["crs"]
    if wm_proj is None:
       print("WARNING: I can't find a CRS in the weather model file, so I will assume you are using WGS84")
       wm_proj = CRS.from_epsg(4326)
    else:
        wm_proj = CRS.from_wkt(wm_proj.to_wkt())

    # Determine the output grid extent here
    wesn = ll_bounds[2:] + ll_bounds[:2]
    out_snwe = transform_bbox(
        wesn, src_crs=4326, dest_crs=crs
    )

    # Clip output grid to multiples of spacing
    if cube_spacing_m is None:
        with xarray.load_dataset(weather_model_file) as ds:
            xpts = ds.x.values
            ypts = ds.y.values
        cube_spacing_m = np.nanmean([np.nanmean(np.diff(xpts)), np.nanmean(np.diff(ypts))])
        if wm_proj.axis_info[0].unit_name == "degree":
            cube_spacing_m = cube_spacing_m * 1.0e5  # Scale by 100km

    if crs.axis_info[0].unit_name == "degree":
        out_spacing = cube_spacing_m / 1.0e5  # Scale by 100km
    else:
        out_spacing = cube_spacing_m
    out_snwe = clip_bbox(out_snwe, out_spacing)

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
        ifWet, ifHydro = getInterpolators(weather_model_file, "total")

        # Build cube
        wetDelay, hydroDelay = _build_cube(
            xpts, ypts, zpts,
            wm_proj, crs,
            [ifWet, ifHydro])

    else:
        out_type = "slant - raytracing"

        # Get pointwise interpolators
        ifWet, ifHydro = getInterpolators(
            weather_model_file,
            kind="pointwise",
            shared=(nproc > 1),
        )

        # Build cube
        if nproc == 1:
            wetDelay, hydroDelay = _build_cube_ray(
                xpts, ypts, zpts,
                dt, los._file, look_dir,
                wm_proj, crs,
                [ifWet, ifHydro])

        ### Use multi-processing here
        else:
            # Pre-build output arrays

            # Create worker pool

            # Loop over heights
            raise NotImplementedError

    # Write output file
    # Modify this as needed for NISAR / other projects
    ds = xarray.Dataset(
        data_vars=dict(
            wet=(["z", "y", "x"],
                 wetDelay,
                 {"units" : "m",
                  "description": f"wet {out_type} delay",
                  "grid_mapping": "cube_projection",
                 }),
            hydro=(["z", "y", "x"],
                   hydroDelay,
                   {"units": "m",
                    "description": f"hydrostatic {out_type} delay",
                    "grid_mapping": "cube_projection",
                   }),
        ),
        coords=dict(
            x=(["x"], xpts),
            y=(["y"], ypts),
            z=(["z"], zpts),
        ),
        attrs=dict(
            Conventions="CF-1.7",
            title="RAiDER geo cube",
            source=os.path.basename(weather_model_file),
            history=str(datetime.datetime.utcnow()) + " RAiDER",
            description=f"RAiDER geo cube - {out_type}",
            reference_time=str(dt),
        ),
    )

    # Write projection system mapping
    ds["cube_projection"] = int()
    for k, v in crs.to_cf().items():
        ds.cube_projection.attrs[k] = v

    # Write z-axis information
    ds.z.attrs["axis"] = "Z"
    ds.z.attrs["units"] = "m"
    ds.z.attrs["description"] = "height above ellipsoid"

    # If in degrees
    if crs.axis_info[0].unit_name == "degree":
        ds.y.attrs["units"] = "degrees_north"
        ds.y.attrs["standard_name"] = "latitude"
        ds.y.attrs["long_name"] = "latitude"

        ds.x.attrs["units"] = "degrees_east"
        ds.x.attrs["standard_name"] = "longitude"
        ds.x.attrs["long_name"] = "longitude"

    else:
        ds.y.attrs["axis"] = "Y"
        ds.y.attrs["standard_name"] = "projection_y_coordinate"
        ds.y.attrs["long_name"] = "y-coordinate in projected coordinate system"
        ds.y.attrs["units"] = "m"

        ds.x.attrs["axis"] = "X"
        ds.x.attrs["standard_name"] = "projection_x_coordinate"
        ds.x.attrs["long_name"] = "x-coordinate in projected coordinate system"
        ds.x.attrs["units"] = "m"

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
    xpts, ypts, zpts, ref_time, orbit_file, look_dir, model_crs,
    pts_crs, interpolators, outputArrs=None
):
    """
    Iterate over interpolators and build a cube using raytracing
    """

    # Some constants for this module
    # TODO - Read this from constants or configuration
    MAX_SEGMENT_LENGTH = 1000.
    MAX_TROPO_HEIGHT = 50000.


    # First load the state vectors into an isce orbit
    orb = isce.core.Orbit([
        isce.core.StateVector(
            isce.core.DateTime(row[0]),
            row[1:4], row[4:7]
        ) for row in np.stack(
            get_sv(orbit_file, ref_time, pad=600), axis=-1
        )
    ])

    # ISCE3 data structures
    elp = isce.core.Ellipsoid()
    dop = isce.core.LUT2d()
    if look_dir.lower() == "right":
        look = isce.core.LookSide.Right
    elif look_dir.lower() == "left":
        look = isce.core.LookSide.Left
    else:
        raise RuntimeError(f"Unknown look direction: {look_dir}")
    logger.debug(f"Look direction: {look_dir}")

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

        xyz = np.stack(
            lla2ecef(llh[1], llh[0], np.full(yy.shape, ht)),
            axis=-1)

        llh[0] = np.deg2rad(llh[0])
        llh[1] = np.deg2rad(llh[1])

        # Step 2 - get LOS vectors for targets
        # TODO - Modify when isce3 vectorization is available
        los = np.full(xyz.shape, np.nan)
        for ii in range(yy.shape[0]):
            for jj in range(yy.shape[1]):
                inp = np.array([
                    llh[0][ii, jj],
                    llh[1][ii, jj],
                    ht])
                inp_xyz = xyz[ii, jj, :]

                if any(np.isnan(inp)) or any(np.isnan(inp_xyz)):
                    continue

                # Local normal vector
                nv = elp.n_vector(inp[0], inp[1])

                # Wavelength does not matter for
                try:
                    aztime, slant_range = isce.geometry.geo2rdr(
                        inp, elp, orb, dop, 0.06, look,
                        threshold=1.0e-7,
                        maxiter=30,
                        delta_range=10.0)
                    sat_xyz, _ = orb.interpolate(aztime)
                    los[ii, jj, :] = (sat_xyz - inp_xyz) / slant_range
                except Exception as e:
                    los[ii, jj, :] = np.nan

        # Free memory here
        llh = None

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
                low_xyz = getTopOfAtmosphere(xyz, los, low_ht, factor=cos_factor)

            # Compute high_xyz
            high_xyz = getTopOfAtmosphere(xyz, los, high_ht, factor=cos_factor)

            # Compute ray length
            ray_length =  np.linalg.norm(high_xyz - low_xyz, axis=-1)

            # Compute cos_factor for first iteration
            if cos_factor is None:
                cos_factor = (high_ht - low_ht) / ray_length

            # Determine number of parts to break ray into
            try:
                nParts = int(np.ceil(ray_length.max() / MAX_SEGMENT_LENGTH)) + 1
            except ValueError:
                raise ValueError(
                    "geo2rdr did not converge. Check orbit coverage"
                )

            if (nParts == 1):
                raise RuntimeError(
                    "Ray with one segment encountered"
                )

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
