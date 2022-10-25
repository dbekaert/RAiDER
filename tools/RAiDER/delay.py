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
from scipy.interpolate import RegularGridInterpolator as Interpolator

from RAiDER.constants import _STEP
from RAiDER.delayFcns import (
    getInterpolators,
    calculate_start_points,
    get_delays,
)
from RAiDER.dem import getHeights
from RAiDER.logger import logger
from RAiDER.losreader import Zenith, Conventional, Raytracing
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import (
    writeDelays, projectDelays, writePnts2HDF5,
    lla2ecef, transform_bbox, clip_bbox,
)


def tropo_delay(args):
    """
    raiderDelay main function.


    Parameters
    ----------
    args: dict  Parameters and inputs needed for processing,
                containing the following key-value pairs:

        los     - tuple, Zenith class object, ('los', 2-band los file), or ('sv', orbit_file)
        lats    - ndarray or str
        lons    - ndarray or str
        heights - see checkArgs for format
        flag    -
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
    lats = args['lats']
    lons = args['lons']
    ll_bounds = args['ll_bounds']
    heights = args['heights']
    flag = args['flag']
    weather_model = args['weather_model']
    wmLoc = args['wmLoc']
    zref = args['zref']
    outformat = args['outformat']
    time = args['times']
    download_only = args['download_only']
    wetFilename = args['wetFilenames']
    hydroFilename = args['hydroFilenames']
    pnts_file = args['pnts_file']
    verbose = args['verbose']

    # logging
    logger.debug('Starting to run the weather model calculation')
    logger.debug('Time type: {}'.format(type(time)))
    logger.debug('Time: {}'.format(time.strftime('%Y%m%d')))
    logger.debug('Flag type is {}'.format(flag))
    logger.debug('DEM/height type is "{}"'.format(heights[0]))
    logger.debug('Max integration height is {:1.1f} m'.format(zref))

    ###########################################################
    # weather model calculation
    useWeatherNodes = flag == 'bounding_box'
    delayType = ["Zenith" if los is Zenith else "LOS"]

    logger.debug('Beginning weather model pre-processing')
    logger.debug('Download-only is {}'.format(download_only))

    weather_model_file = prepareWeatherModel(
        weather_model,
        time,
        wmLoc=wmLoc,
        lats=ll_bounds[:2] if isinstance(lats, str) else lats,
        lons=ll_bounds[2:] if isinstance(lons, str) else lons,
        zref=zref,
        download_only=download_only,
        makePlots=verbose,
    )

    if download_only:
        return None, None
    elif useWeatherNodes:
        if heights[0] == 'lvs':
            # compute delays at the correct levels
            ds = xarray.load_dataset(weather_model_file)
            ds['wet_total'] = ds['wet_total'].interp(z=heights[1])
            ds['hydro_total'] = ds['hydro_total'].interp(z=heights[1])
        else:
            logger.debug(
                'Only Zenith delays at the weather model nodes '
                'are requested, so I am exiting now. Delays have '
                'been written to the weather model file; see '
                '{}'.format(weather_model_file)
            )
        return None, None

    ###########################################################
    # Load the downloaded model file
    ds = xarray.load_dataset(weather_model_file)
    try:
        wm_proj = ds['CRS']
    except:
        print("WARNING: I can't find a CRS in the weather model file, so I will assume you are using WGS84")
        wm_proj = 4326
    ####################################################################

    ####################################################################
    # Calculate delays
    if isinstance(los, (Zenith, Conventional)):
        # Start actual processing here
        logger.debug("Beginning DEM calculation")
        # Lats, Lons will be translated from file to array here if needed
        lats, lons, hgts = getHeights(lats, lons, heights, useWeatherNodes)
        logger.debug(
            'DEM height range for the queried region is %.2f-%.2f m',
            np.nanmin(hgts), np.nanmax(hgts)
        )
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

    if heights[0] == 'lvs':
        outName = wetFilename[0].replace('wet', 'delays')
        writeDelays(
            flag,
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

        writeDelays(flag, wetDelay, hydroDelay, lats, lons,
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
    download_only   - Only download the raw weather model data
    filename        - Output filename
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
    logger.debug(f'Download_only is {args["download_only"]}')

    weather_model_file = prepareWeatherModel(
        args["weather_model"],
        args["time"],
        wmLoc=args["wmLoc"],
        lats=args["ll_bounds"][:2],
        lons=args["ll_bounds"][2:],
        zref=args["zref"],
        download_only=args["download_only"],
        makePlots=args["verbose"],
    )

    if args["download_only"]:
        return None, None

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

    # Get ZTD interpolators
    ifWet, ifHydro = getInterpolators(weather_model_file, "total")

    # Output grid points - North up grid
    zpts = np.array(args["heights"])
    xpts = np.arange(out_snwe[2], out_snwe[3] + spacing, spacing)
    ypts =np.arange(out_snwe[1], out_snwe[0] - spacing, -spacing)
    xx, yy = np.meshgrid(xpts, ypts)
    # Output arrays
    wetDelay = np.zeros((zpts.size, ypts.size, xpts.size))
    hydroDelay = np.zeros(wetDelay.shape)

    # Loop over heights and compute delays
    for ii, ht in enumerate(zpts):
        # pts is in weather model system
        if wm_proj != args["crs"]:
            pts = transformPoints(
                yy, xx, np.full(yy.shape, ht),
                args["crs"], wm_proj
            )
        else:
            pts = np.stack([yy, xx, np.full(yy.shape, ht)], axis=-1)

        wetDelay[ii,...] = ifWet(pts)
        hydroDelay[ii,...] = ifHydro(pts)

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
