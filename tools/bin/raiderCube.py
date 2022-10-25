#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import argparse
import copy
from dateutil.parser import parse as parsedt
from textwrap import dedent
import numpy as np
import pyproj

from RAiDER.cli.parser import add_bbox, add_out, add_verbose
from RAiDER.models.allowed import ALLOWED_MODELS
from RAiDER.logger import logger, logging
from RAiDER.utilFcns import rio_profile, rio_extents, transform_bbox
from RAiDER.checkArgs import modelName2Module
from RAiDER.delay import tropo_delay_cube


def create_parser():
    """
    Create command line parser for this executable
    """

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent("""\
            Calculate tropospheric delay cube from a weather model.
            Usage examples:
            raiderDelay.py --datetime 20200103T23:00:00 -b 39 40 -79 -78
                           --model ERA5 --zref 15000 --heighlvs 800 850 900
            """)
    )

    datetimes = p.add_argument_group('Datetime')
    datetimes.add_argument(
        '--datetime', dest='datetimeList',
        help="Space separated time tags",
        nargs="+",
        type=str,
        required=True
    )

    # Area
    area = p.add_argument_group('Area of Interest (Supply one)').add_mutually_exclusive_group(required=True)
    add_bbox(area)
    area.add_argument(
        '--product',
        default=None,
        type=str,
        dest='geocoded_product',
        help='Geocoded product to extract bounds information from'
    )

    # Cube
    cube = p.add_argument_group('Output cube parameters').add_mutually_exclusive_group(required=True)

    cube.add_argument(
        '--heightlvs',
        help=("""A space-deliminited list of heights"""),
        default=np.arange(-500., 9100., 500.).tolist(),
        nargs='+',
        type=float,
        required=False)
    cube.add_argument(
        '--spacinginm',
        help="Horizontal spacing in meters",
        default=2000.,
        type=float,
        required=False)
    cube.add_argument(
        '--epsg',
        help="EPSG code for output cube",
        default=None,
        type=int)

    # Weather model
    weather = p.add_argument_group("Weather model. See documentation for details")
    weather.add_argument(
        '--model',
        help="Weather model option to use.",
        type=lambda s: s.upper().replace("-", ""),
        choices=ALLOWED_MODELS,
        default='ERA5T')
    weather.add_argument(
        '--files',
        help="""OUT/PLEV or HDF5 file(s) """,
        default=None, nargs='+', type=str, metavar="FILES")
    weather.add_argument(
        '--weatherFiles', '-w',
        help='Directory location of/to write weather model files',
        default=None, dest='wmLoc')

    # Add parameters
    misc = p.add_argument_group("Run parameters")
    misc.add_argument(
        '--zref', '-z',
        help='Height limit when integrating (meters) (default: 15000 m)',
        type=float,
        default=15000.)
    misc.add_argument(
        '--outformat',
        help='GDAL-compatible file format if surface delays are requested.',
        type= lambda s: s.lower(),
        choices=["nc", "h5"],
        default="nc")
    add_out(misc)
    misc.add_argument(
        '--download_only',
        help='Download weather model only without processing? Default False',
        action='store_true', dest='download_only', default=False)
    add_verbose(misc)

    return p


def checkArgs(args, p):
    """
    Validate input arguments
    """
    # Check datetimes
    args.datetimeList = [parsedt(x) for x in args.datetimeList]

    # Check height levels
    if args.wmLoc is not None:
        wmLoc = args.wmLoc
    else:
        wmLoc = os.path.join(args.out, 'weather_files')

    if not os.path.exists(wmLoc):
        os.mkdir(wmLoc)

    # Set bounding box here
    if args.query_area is None and args.geocoded_product is None:
        raise ValueError(
            f"Must provide a bounding box or a geocoded product as AOI"
        )

    if args.query_area is None:
        profile = rio_profile(args.geocoded_product)
        extent = rio_extents(rio_profile)

        # Overwrite bbox with estimated bbox
        # extent is WESN and we want SNWE for parser compatibility
        args.query_area = transform_bbox(extent, src_crs=profile["crs"],
                                         dest_crs=4326)

    if (np.min(args.query_area[:2]) < -90) | (np.max(args.query_area[:2]) > 90):
        raise ValueError('Lats are out of N/S bounds; are your lat/lon coordinates switched?')

    # Determine output projection system
    if args.epsg is not None:
        args.epsg = pyproj.CRS.from_epsg(args.epsg)
    elif args.geocoded_product is not None:
        args.epsg = profile["crs"]
    else:
        # TODO - Handle dateline crossing
        mid_lon = np.mean(args.query_area[:2])
        args.epsg = pyproj.CRS.from_epsg(
            np.flooor((mid_lon+180.)/6.0).astype(np.int) + 1
        )
        logger.info(f"Determining EPSG code from bounds: {args.epsg}")

    # Load weather model
    try:
        _, model_obj = modelName2Module(args.model)
    except ModuleNotFoundError:
        raise NotImplementedError(
            dedent('''
                Model {} is not yet fully implemented,
                please contribute!
                '''.format(args.model))
        )
    if args.model in ['WRF', 'HDF5'] and args.files is None:
        raise RuntimeError(
            'Argument --files is required with model {}'.format(args.model)
        )

    weathers = {
        'type': model_obj(),
        'files': args.files,
        'name': args.model.lower().replace('-', '')
    }

    # Handling output
    if args.out is None:
       args.out = os.getcwd()

    filenames = []
    for time in args.datetimeList:
        filenames.append(
            makeDelayFileName(time, args.outformat, args.model, args.out)
        )

    # Gather output in one dictionary
    outArgs = {}
    outArgs["ll_bounds"] = args.query_area
    outArgs["heights"] = args.heightlvs
    outArgs["weather_model"] = weathers
    outArgs["wmLoc"] = wmLoc
    outArgs["zref"] = args.zref
    outArgs["spacinginm"] = args.spacinginm
    outArgs["crs"] = args.epsg
    outArgs["outformat"] = args.outformat
    outArgs["time"] = args.datetimeList
    outArgs["download_only"] = args.download_only
    outArgs["out"] = args.out
    outArgs["verbose"] = args.verbose
    outArgs["filename"] = filenames

    return outArgs


def makeDelayFileName(time, outformat, model_name, out):
    '''
    return name for the netcdf or h5 file
    '''
    filename = f"{model_name}_tropo_{time:%Y%m%dT%H%M%S}_zstd.{outformat}"
    return os.path.join(out, filename)


def parseCMD():
    """
    Parse command-line arguments and pass to generate_cube
    We'll parse arguments and call geocube.py
    """

    p = create_parser()
    args = p.parse_args()

    # Argument checking
    args = checkArgs(args, p)

    if args["verbose"]:
        logger.setLevel(logging.DEBUG)

    for time, outname in zip(args["time"], args["filename"]):
        try:
            args_copy = copy.deepcopy(args)
            args_copy["time"] = time
            args_copy["filename"] = outname

            tropo_delay_cube(args_copy)
        except RuntimeError:
            logger.exception(f"Datetime {time} failed")
            continue


if __name__ == '__main__':
    parseCMD()
