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
    lla2ecef, transform_bbox, clip_bbox,
)


def tropo_delay(dt, wf, hf, args):
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
    wetFilename = wf
    hydroFilename = hf

    los = args['los']
    ll_bounds = args['bounding_box']
    heights = args['dem']
    weather_model = args['weather_model']
    wmLoc = args['weather_model_directory']
    zref = args['zref']
    outformat = args['raster_format']
    download_only = False
    verbose = args['verbose']
    aoi = args['aoi']

    # logging
    logger.debug('Starting to run the weather model calculation')
    logger.debug('Time type: {}'.format(type(dt)))
    logger.debug('Time: {}'.format(dt.strftime('%Y%m%d')))
    logger.debug('Max integration height is {:1.1f} m'.format(zref))

    ###########################################################
    # weather model calculation
    delayType = ["Zenith" if los is Zenith else "LOS"]

    logger.debug('Beginning weather model pre-processing')
    logger.debug('Download-only is {}'.format(download_only))

    if ll_bounds is None:
        ll_bounds = aoi.bounds()

    weather_model_file = prepareWeatherModel(
        weather_model,
        dt,
        wmLoc=wmLoc,
        ll_bounds=ll_bounds,
        zref=zref,
        download_only=download_only,
        makePlots=verbose,
    )

    if aoi.type() == 'bounding_box':
        if args['height_levels'] is not None:
            # compute delays at the correct levels
            ds = xarray.load_dataset(weather_model_file)
            ds['wet_total'] = ds['wet_total'].interp(z=args['height_levels'])
            ds['hydro_total'] = ds['hydro_total'].interp(z=args['height_levels'])
        else:
            logger.debug(
                'Only Zenith delays at the weather model nodes '
                'are requested, so I am exiting now. Delays have '
                'been written to the weather model file; see '
                '{}'.format(weather_model_file)
            )
        return None, None

    ###########################################################
    # Load the downloaded model file for CRS information
    ds = xarray.load_dataset(weather_model_file)
    try:
        wm_proj = ds['CRS']
    except:
        print("WARNING: I can't find a CRS in the weather model file, so I will assume you are using WGS84")
        wm_proj = 4326
    ds.close()
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
                wm_proj
            )
        else:
            # interpolators require y, x, z
            pnts = np.stack([lats, lons, hgts], axis=-1)

        # either way I'll need the ZTD
        ifWet, ifHydro = getInterpolators(weather_model_file, 'total')
        wetDelay = ifWet(pnts)
        hydroDelay = ifHydro(pnts)

        # TODO - handle this better during parsing
        los._pad = 600
        los._time = dt

        # return the delays (ZTD or STD)
        wetDelay = los(wetDelay)
        hydroDelay = los(hydroDelay)

    elif isinstance(los, Raytracing):
        raise NotImplementedError
    else:
        raise ValueError("Unknown operation type")

    del ds  # cleanup

    ###########################################################
    # Write the delays to file
    # Different options depending on the inputs

    if heights is not None:
        outName = wetFilename[0].replace('wet', 'delays')
        writeDelays(
            aoi.type(),
            wetDelay,
            hydroDelay,
            lats,
            lons,
            outName,
            zlevels=hgts,
            outformat=outformat,
            delayType=delayType
        )
        logger.info('Finished writing data to %s', outName)

    else:
        if not isinstance(wetFilename, str):
            wetFilename = wetFilename[0]
            hydroFilename = hydroFilename[0]

        writeDelays(aoi.type(), wetDelay, hydroDelay, lats, lons,
                    wetFilename, hydroFilename, outformat=outformat,
                    proj=None, gt=None, ndv=0.)
        logger.info('Finished writing data to %s', wetFilename)

    return wetDelay, hydroDelay


def tropo_delay_cube(args):
    """
    raiderDelay cube generation function.
    Parameters
    ----------
    ll_bounds       - Lat/Lon bbox
    heights         - Height levels for output cube
    weather_model   - Type of weather model to use
    wmLoc           - Directory containing weather model files
    zref            - max integration height
    spacinginm      - output cube spacing in meters
    crs             - output cube pyproj crs
    out             - output directory
    outformat       - File format to use for raster outputs
    time            - Datetime to calculate delay
    filename        - Output filename
    orbit           - Orbit filename
    verbose         - Verbose printing
    """

    # logging
    logger.debug('Starting to run the weather model cube calculation')
    logger.debug(f'Time: {args["time"]}')
    logger.debug(f'Output height range is {min(args["heights"])} to {max(args["heights"])}')
    logger.debug(f'Max integration height is {args["zref"]:1.1f} m')
    logger.debug(f'Output cube projection is {args["crs"].to_wkt()}')
    logger.debug(f'Output cube spacing is {args["spacinginm"]} m')

    # We are using zenith model only for now
    logger.debug('Beginning weather model pre-processing')

    weather_model_file = prepareWeatherModel(
        args["weather_model"],
        args["time"],
        wmLoc=args["wmLoc"],
        lats=args["ll_bounds"][:2],
        lons=args["ll_bounds"][2:],
        zref=args["zref"],
        makePlots=args["verbose"],
    )

    # Determine the output grid extent here
    wesn = args["ll_bounds"][2:] + args["ll_bounds"][:2]
    out_snwe = transform_bbox(
        wesn, src_crs=4326, dest_crs=args["crs"]
    )

    # Clip output grid to multiples of spacing
    # If output is desired in degrees
    if args["crs"].axis_info[0].unit_name == "degree":
        spacing = args["spacinginm"]/ 1.0e5
        out_snwe = clip_bbox(out_snwe, spacing)
    else:
        spacing = args["spacinginm"]
        out_snwe = clip_bbox(out_snwe, spacing)

    # Load downloaded weather model file to get projection info
    ds = xarray.load_dataset(weather_model_file)
    try:
        wm_proj = ds["CRS"]
    except:
        print("WARNING: I can't find a CRS in the weather model file, so I will assume you are using WGS84")
        wm_proj = CRS.from_epsg(4326)
    ds.close()


    # Output grid points - North up grid
    zpts = np.array(args["heights"])
    xpts = np.arange(out_snwe[2], out_snwe[3] + spacing, spacing)
    ypts =np.arange(out_snwe[1], out_snwe[0] - spacing, -spacing)

    # If no orbit is provided
    # Build zenith delay cube
    if args["orbit"] is None:
        # Get ZTD interpolators
        ifWet, ifHydro = getInterpolators(weather_model_file, "total")

        # Build cube
        wetDelay, hydroDelay = build_cube(
            xpts, ypts, zpts,
            wm_proj, args["crs"],
            [ifWet, ifHydro])
    else:
        # Get pointwise interpolators
        ifWet, ifHydro = getInterpolators(weather_model_file, "pointwise")

        # Build cube
        wetDelay, hydroDelay = build_cube_ray(
            xpts, ypts, zpts,
            args["time"], args["orbit"],
            wm_proj, args["crs"],
            [ifWet, ifHydro])

    # Write output file
    # Modify this as needed for NISAR / other projects
    ds = xarray.Dataset(
        data_vars=dict(
            wet=(["z", "y", "x"],
                 wetDelay,
                 {"units" : "m",
                  "description": "wet zenith delay",
                  "grid_mapping": "cube_projection",
                 }),
            hydro=(["z", "y", "x"],
                   hydroDelay,
                   {"units": "m",
                    "description": "hydrostatic zenith delay",
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
            description="RAiDER geo cube",
            reference_time=str(args["time"]),
        ),
    )

    # Write projection system mapping
    ds["cube_projection"] = int()
    for k, v in args["crs"].to_cf().items():
        ds.cube_projection.attrs[k] = v

    # Write z-axis information
    ds.z.attrs["axis"] = "Z"
    ds.z.attrs["units"] = "m"
    ds.z.attrs["description"] = "height above ellipsoid"

    # If in degrees
    if args["crs"].axis_info[0].unit_name == "degree":
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

    ds.to_netcdf(args["filename"], mode="w")


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
    the array of query points in the weather model coordinate system
    '''
    t = Transformer.from_crs(old_proj, new_proj)
    return np.stack(t.transform(lats, lons, hgts), axis=-1).T


def build_cube(xpts, ypts, zpts, model_crs, pts_crs, interpolators):
    """
    Iterate over interpolators and build a cube
    """
    # Create a regular 2D grid
    xx, yy = np.meshgrid(xpts, ypts)

    # Output arrays
    outputArrs = [np.zeros((zpts.size, ypts.size, xpts.size))
                  for mm in range(len(interpolators))]

    # Loop over heights and compute delays
    for ii, ht in enumerate(zpts):
        # pts is in weather model system
        if model_crs != pts_crs:
            pts = transformPoints(
                yy, xx, np.full(yy.shape, ht),
                pts_crs, model_crs
            )
        else:
            pts = np.stack([yy, xx, np.full(yy.shape, ht)], axis=-1)

        for mm, intp in enumerate(interpolators):
            outputArrs[mm][ii,...] = intp(pts)

    return outputArrs


def build_cube_ray(xpts, ypts, zpts, ref_time, orbit_file, model_crs, pts_crs, interpolators):
    """
    Iterate over interpolators and build a cube
    """
    # Some constants for this module
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
    # TODO - Assuming right-looking for now
    elp = isce.core.Ellipsoid()
    dop = isce.core.LUT2d()
    look = isce.core.LookSide.Right

    # Get model heights in an array
    # Assumption: All interpolators here are on the same grid
    model_zs = interpolators[0].grid[2]

    # Create a regular 2D grid
    xx, yy = np.meshgrid(xpts, ypts)

    # Output arrays
    outputArrs = [np.zeros((zpts.size, ypts.size, xpts.size))
                  for mm in range(len(interpolators))]

    # Various transformers needed here
    epsg4326 = CRS.from_epsg(4326)
    cube_to_llh = Transformer.from_crs(pts_crs, epsg4326,
                                       always_xy=True)
    ecef_to_model = Transformer.from_crs(CRS.from_epsg(4978), model_crs)

    # Loop over heights of output cube and compute delays
    for hh, ht in enumerate(zpts):
        # Slices to fill on output
        outSubs = [x[hh, ...] for x in outputArrs]

        # Step 1:  transform points to llh and xyz
        if pts_crs != epsg4326:
            llh = cube_to_llh.transform(xx, yy, np.full(yy.shape, ht))
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
                low_xyz = getTopOfAtmosphere(xyz, los, low_ht)

            # Compute high_xyz
            high_xyz = getTopOfAtmosphere(xyz, los, high_ht)

            # Compute ray length
            ray_length =  np.linalg.norm(high_xyz - low_xyz, axis=-1)

            # Determine number of parts to break ray into
            nParts = int(np.ceil(ray_length.max() / MAX_SEGMENT_LENGTH)) + 1
            if (nParts == 1):
                1/0

            # fractions
            fracs = np.linspace(0., 1., num=nParts)

            # Integrate over the ray
            for findex, ff in enumerate(fracs):
                # Ray point in ECEF coordinates
                pts_xyz = low_xyz + ff * (high_xyz - low_xyz)

                # Ray point in model coordinates
                pts = np.stack(
                    ecef_to_model.transform(
                        pts_xyz[..., 0],
                        pts_xyz[..., 1],
                        pts_xyz[..., 2]
                    ), axis=-1)

                # Trapezoidal integration with scaling
                wt = 0.5 if findex in [0, fracs.size-1] else 1.0
                wt *= ray_length *1.0e-6 / (nParts - 1.0)

                # For each interpolator, integrate between levels
                for mm, out in enumerate(outSubs):
                    val =  interpolators[mm](pts)

                    # TODO - This should not occur if there is enough padding in model
                    val[np.isnan(val)] = 0.0
                    out += wt * val


    return outputArrs
