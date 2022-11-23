# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson, David Bekaert & Yang Lei
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os

import datetime
import h5py
import numpy as np
import xarray
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

import isce3.ext.isce3 as isce
from RAiDER.constants import _STEP
from RAiDER.delayFcns import (
    getInterpolators,
    calculate_start_points,
    get_delays,
)
from RAiDER.dem import getHeights
from RAiDER.logger import logger
from RAiDER.llreader import BoundingBox
from RAiDER.losreader import Zenith, Conventional, Raytracing, get_sv, getTopOfAtmosphere
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import (
    writeDelays, projectDelays, writePnts2HDF5,
    lla2ecef, transform_bbox, clip_bbox, rio_profile,
)


def tropo_delay(dt, wetFilename, hydroFilename, args):
    """
    raiderDelay main function.

    Parameterss
    ----------
    args: dict  Parameters and inputs needed for processing,
                containing the following key-value pairs:

        los     - tuple, Zenith class object, ('los', 2-band los file), or ('sv', orbit_file)
        lats    - ndarray or str
        lons    - ndarray or str
        heights - see checkArgs for format
        weather_model   - type of weather model to use
        wmLoc   - Directory containing weather model files
        zref    - max integration height
        outformat       - File format to use for raster outputs
        time    - list of datetimes to calculate delays
        download_only   - Only download the raw weather model data and exit
        wetFilename     -
        hydroFilename   -
        pnts_file       - Input a points file from previous run
        verbose - verbose printing
    """
    # unpacking the dictionairy
    los = args['los']
    heights = args['dem']
    weather_model = args['weather_model']
    wmLoc = args['weather_model_directory']
    zref = args['zref']
    outformat = args['raster_format']
    verbose = args['verbose']
    aoi   = args['aoi']

    steps = args['runSteps']
    download_only = True if len(steps) == 1 and \
                        steps[0] == 'load_weather_model' else False

    if los.ray_trace():
        ll_bounds = aoi.add_buffer(buffer=1) # add a buffer for raytracing
    else:
        ll_bounds = aoi.bounds()

    # logging
    logger.debug('Starting to run the weather model calculation')
    logger.debug('Time type: {}'.format(type(dt)))
    logger.debug('Time: {}'.format(dt.strftime('%Y%m%d')))
    logger.debug('Max integration height is {:1.1f} m'.format(zref))

    ###########################################################
    # weather model calculation
    delayType = ["Zenith" if los is Zenith else "LOS"]

    logger.debug('Beginning weather model pre-processing')

    weather_model_file = prepareWeatherModel(
        weather_model,
        dt,
        wmLoc=wmLoc,
        ll_bounds=ll_bounds, # SNWE
        zref=zref,
        download_only=download_only,
        makePlots=verbose,
    )

    if download_only:
        logger.debug('Weather model has downloaded. Finished.')
        return None, None


    if aoi.type() == 'bounding_box' or \
                (args['height_levels'] and aoi.type() != 'station_file'):
        # This branch is specifically for cube generation
        try:
            tropo_delay_cube(
                dt, wetFilename, args,
                model_file=weather_model_file,
            )
        except Exception as e:
            logger.error(e)
            raise RuntimeError('Something went wrong in calculating delays on the cube')
        return None, None

    ###########################################################
    # Load the downloaded model file for CRS information
    wm_proj = rio_profile(f"netcdf:{weather_model_file}:t")[0]["crs"]
    if wm_proj is None:
        print("WARNING: I can't find a CRS in the weather model file, so I will assume you are using WGS84")
        wm_proj = CRS.from_epsg(4326)
    else:
        wm_proj = CRS.from_wkt(wm_proj.to_wkt())
    ####################################################################

    ####################################################################
    # Calculate delays
    if isinstance(los, (Zenith, Conventional)):
        # Start actual processing here
        logger.debug("Beginning DEM calculation")
        # Lats, Lons will be translated from file to array here if needed

        # read lats/lons
        lats, lons = aoi.readLL()
        hgts = aoi.readZ()

        los.setPoints(lats, lons, hgts)

        # Transform query points if needed
        pnt_proj = CRS.from_epsg(4326)
        if wm_proj != pnt_proj:
            pnts = transformPoints(
                lats,
                lons,
                hgts,
                pnt_proj,
                wm_proj,
            )
        else:
            # interpolators require y, x, z
            pnts = np.stack([lats, lons, hgts], axis=-1)

        # either way I'll need the ZTD
        ifWet, ifHydro = getInterpolators(weather_model_file, 'total')
        wetDelay = ifWet(pnts)
        hydroDelay = ifHydro(pnts)

        # return the delays (ZTD or STD)
        wetDelay = los(wetDelay)
        hydroDelay = los(hydroDelay)

    elif isinstance(los, Raytracing):
        raise NotImplementedError
    else:
        raise ValueError("Unknown operation type")

    ###########################################################
    # Write the delays to file
    # Different options depending on the inputs

    if not isinstance(wetFilename, str):
        wetFilename   = wetFilename[0]
        hydroFilename = hydroFilename[0]

    if heights is not None:
        writeDelays(
            aoi,
            wetDelay,
            hydroDelay,
            lats,
            lons,
            wetFilename,
            hydroFilename,
            zlevels=hgts,
            outformat=outformat,
            delayType=delayType
        )
        logger.info('Finished writing data to %s', wetFilename)

    else:

        if aoi.type() == 'station_file':
            wetFilename = f'{os.path.splitext(wetFilename)[0]}.csv'

        writeDelays(aoi, wetDelay, hydroDelay, lats, lons,
                    wetFilename, hydroFilename, outformat=outformat,
                    proj=None, gt=None, ndv=0.)

        logger.info('Finished writing data to %s', wetFilename)

    return wetDelay, hydroDelay


def tropo_delay_cube(dt, wf, args, model_file=None):
    """
    raiderDelay cube generation function.

    Same as tropo_delay() above.
    """
    los = args['los']
    weather_model = args['weather_model']
    wmLoc = args['weather_model_directory']
    zref = args['zref']
    download_only = False
    verbose = args['verbose']
    aoi = args['aoi']
    cube_spacing = args["cube_spacing_in_m"]
    ll_bounds = aoi.bounds()

    try:
        crs  = CRS(args['output_projection'])
    except CRSError:
        raise ValueError('output_projection argument is not a valid CRS specifier')

    # For testing multiprocessing
    # TODO - move this to configuration
    nproc = 1

    # logging
    logger.debug('Starting to run the weather model cube calculation')
    logger.debug(f'Time: {dt}')
    logger.debug(f'Max integration height is {zref:1.1f} m')
    logger.debug(f'Output cube projection is {crs.to_wkt()}')
    logger.debug(f'Output cube spacing is {cube_spacing} m')

    # We are using zenith model only for now
    logger.debug('Beginning weather model pre-processing')

    # If weather model file is not provided
    if model_file is None:
        weather_model_file = prepareWeatherModel(
            weather_model,
            dt,
            wmLoc=wmLoc,
            ll_bounds=ll_bounds,
            zref=zref,
            download_only=download_only,
            makePlots=verbose
        )
    else:
        weather_model_file = model_file

    # Determine the output grid extent here
    wesn = ll_bounds[2:] + ll_bounds[:2]
    out_snwe = transform_bbox(
        wesn, src_crs=4326, dest_crs=crs
    )

    # Clip output grid to multiples of spacing
    # If output is desired in degrees
    if crs.axis_info[0].unit_name == "degree":
        out_spacing = cube_spacing / 1.0e5  # Scale by 100km
        out_snwe = clip_bbox(out_snwe, out_spacing)
    else:
        out_spacing = cube_spacing
        out_snwe = clip_bbox(out_snwe, out_spacing)

    logger.debug(f"Output SNWE: {out_snwe}")
    logger.debug(f"Output cube spacing: {out_spacing}")

    # Load downloaded weather model file to get projection info
    with xarray.load_dataset(weather_model_file) as ds:
        # Output grid points - North up grid
        if args['height_levels'] is not None:
            heights = args['height_levels']
        else:
            heights = ds.z.values

    logger.debug(f'Output height range is {min(heights)} to {max(heights)}')

    # Load CRS from weather model file
    wm_proj = rio_profile(f"netcdf:{weather_model_file}:t")[0]["crs"]
    if wm_proj is None:
       print("WARNING: I can't find a CRS in the weather model file, so I will assume you are using WGS84")
       wm_proj = CRS.from_epsg(4326)
    else:
        wm_proj = CRS.from_wkt(wm_proj.to_wkt())

    # Build the output grid
    zpts = np.array(heights)
    xpts = np.arange(out_snwe[2], out_snwe[3] + out_spacing, out_spacing)
    ypts = np.arange(out_snwe[1], out_snwe[0] - out_spacing, -out_spacing)

    # If no orbit is provided
    # Build zenith delay cube
    if los.is_Zenith():
        out_type = "zenith"
        out_filename = wf.replace("wet", "tropo")

        # Get ZTD interpolators
        ifWet, ifHydro = getInterpolators(weather_model_file, "total")

        # Build cube
        wetDelay, hydroDelay = build_cube(
            xpts, ypts, zpts,
            wm_proj, crs,
            [ifWet, ifHydro])

    else:
        out_type = "slant range"
        if not los.ray_trace():
            out_filename = wf.replace("_ztd", "_std").replace("wet", "tropo")
        else:
            out_filename = wf.replace("_ztd", "_ray").replace("wet", "tropo")

        if args["look_dir"].lower() not in ["right", "left"]:
            raise ValueError(
                f'Unknown look direction: {args["look_dir"]}'
            )

        # Get pointwise interpolators
        ifWet, ifHydro = getInterpolators(
            weather_model_file,
            kind="pointwise",
            shared=(nproc > 1),
        )

        if los.ray_trace():
            # Build cube
            if nproc == 1:
                wetDelay, hydroDelay = build_cube_ray(
                    xpts, ypts, zpts,
                    dt, args["los"]._file, args["look_dir"],
                    wm_proj, crs,
                    [ifWet, ifHydro])

            ### Use multi-processing here
            else:
                # Pre-build output arrays

                # Create worker pool

                # Loop over heights
                raise NotImplementedError
        else:
            raise NotImplementedError('Conventional STD is not yet implemented on the cube')

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
            reference_time=str(args["time"]),
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


    ext = os.path.splitext(out_filename)
    if ext not in '.nc .h5'.split():
        out_filename = f'{os.path.splitext(out_filename)[0]}.nc'
        logger.debug('Invalid extension %s for cube. Defaulting to .nc', ext)

    if out_filename.endswith(".nc"):
        ds.to_netcdf(out_filename, mode="w")
    elif out_filename.endswith(".h5"):
        ds.to_netcdf(out_filename, engine="h5netcdf", invalid_netcdf=True)

    logger.info('Finished writing data to: %s', out_filename)
    return


def checkQueryPntsFile(pnts_file, query_shape):
    '''
    Check whether the query points file exists, and if it
    does, check that the shapes are all consistent
    '''
    write_flag = True
    if os.path.exists(pnts_file):
        # Check whether the number of points is consistent with the new inputs
        with h5py.File(pnts_file, 'r') as f:
            if query_shape == tuple(f['lon'].attrs['Shape']):
                write_flag = False

    return write_flag


def transformPoints(lats, lons, hgts, old_proj, new_proj):
    '''
    Transform lat/lon/hgt data to an array of points in a new
    projection

    Parameters
    ----------
    lats - WGS-84 latitude (EPSG: 4326)
    lons - ditto for longitude
    hgts - Ellipsoidal height in meters
    old_proj - the original projection of the points
    new_proj - the new projection in which to return the points

    Returns
    -------
    the array of query points in the weather model coordinate system (YX)
    '''
    t = Transformer.from_crs(old_proj, new_proj)

    # Flags for flipping inputs or outputs
    in_flip = old_proj.axis_info[0].direction == "east"
    out_flip = new_proj.axis_info[0].direction == "east"

    if in_flip:
        res = t.transform(lons, lats, hgts)
    else:
        res = t.transform(lats, lons, hgts)

    if out_flip:
        return np.stack((res[1], res[0], res[2]), axis=-1).T
    else:
        return np.stack(res, axis=-1).T


def build_cube(xpts, ypts, zpts, model_crs, pts_crs, interpolators):
    """
    Iterate over interpolators and build a cube
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


def build_cube_ray(xpts, ypts, zpts, ref_time, orbit_file, look_dir, model_crs,
                   pts_crs, interpolators, outputArrs=None):
    """
    Iterate over interpolators and build a cube
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
