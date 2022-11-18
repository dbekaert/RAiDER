# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os

import pandas as pd

from datetime import datetime

from RAiDER.losreader import Zenith
from RAiDER.llreader import BoundingBox


def checkArgs(args):
    '''
    Helper fcn for checking argument compatibility and returns the
    correct variables
    '''

    #########################################################################################################################
    # Directories
    if not os.path.exists(args.weather_model_directory):
        os.mkdir(args.weather_model_directory)

    #########################################################################################################################
    # Date and Time parsing
    args.date_list = [datetime.combine(d, args.time) for d in args.date_list]

    #########################################################################################################################
    # LOS finalizing
    args.los.setLookDir(args['look_dir'])

    #########################################################################################################################
    # filenames
    wetNames, hydroNames = [], []
    for d in args.date_list:
        if not args.aoi is not BoundingBox:
            if args.station_file is not None:
                wetFilename = os.path.join(
                    args.output_directory,
                    '{}_Delay_{}.csv'
                    .format(
                        args.weather_model,
                        args.time.strftime('%Y%m%dT%H%M%S'),
                    )
                )
                hydroFilename = wetFilename

                # copy the input file to the output location for editing
                indf = pd.read_csv(args.query_area).drop_duplicates(subset=["Lat", "Lon"])
                indf.to_csv(wetFilename, index=False)

            else:
                wetNames.append(None)
                hydroNames.append(None)
        else:
            wetFilename, hydroFilename = makeDelayFileNames(
                d,
                args.los,
                args.raster_format,
                args.weather_model._dataset.upper(),
                args.output_directory,
            )

            wetNames.append(wetFilename)
            hydroNames.append(hydroFilename)

    args.wetFilenames = wetNames
    args.hydroFilenames = hydroNames

    return args


def makeDelayFileNames(time, los, outformat, weather_model_name, out):
    '''
    return names for the wet and hydrostatic delays.

    # Examples:
    >>> makeDelayFileNames(datetime(2020, 1, 1, 0, 0, 0), None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_00_00_00_ztd.h5', 'some_dir/model_name_hydro_00_00_00_ztd.h5')
    >>> makeDelayFileNames(None, None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_ztd.h5', 'some_dir/model_name_hydro_ztd.h5')
    '''
    format_string = "{model_name}_{{}}_{time}{los}.{ext}".format(
        model_name=weather_model_name,
        time=time.strftime("%Y%m%dT%H%M%S_") if time is not None else "",
        los="ztd" if (isinstance(los, Zenith) or los is None) else "std",
        ext=outformat
    )
    hydroname, wetname = (
        format_string.format(dtyp) for dtyp in ('hydro', 'wet')
    )

    hydro_file_name = os.path.join(out, hydroname)
    wet_file_name = os.path.join(out, wetname)
    return wet_file_name, hydro_file_name



