# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import rasterio.drivers as rd

from RAiDER.logger import logger
from RAiDER.losreader import Zenith


def checkArgs(args):
    """
    Check argument compatibility and return the correct variables.
    """
    # Directories
    if args.weather_model_directory is None:
        args.weather_model_directory = os.path.join(args.output_directory, 'weather_files')

    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(args.weather_model_directory, exist_ok=True)
    args['weather_model'].set_wmLoc(args.weather_model_directory)

    #########################################################################################################################
    # Date and Time parsing
    args.date_list = [datetime.combine(d, args.time) for d in args.date_list]
    if (len(args.date_list) > 1) & (args.orbit_file is not None):
        logger.warning(
            'Only one orbit file is being used to get the look vectors for all requested times, if you want to use '
            'separate orbit files you will need to run raider separately for each time.'
        )

    args.los.setTime(args.date_list[0])

    #########################################################################################################################
    # filenames
    wetNames, hydroNames = [], []
    for d in args.date_list:
        if args.aoi.type() != 'bounding_box':
            # Handle the GNSS station file
            if args.aoi.type() == 'station_file':
                wetFilename = str(
                    run_config.runtime_group.output_directory /
                    f'{run_config.weather_model._dataset.upper()}_Delay_{d.strftime("%Y%m%dT%H%M%S")}_ztd.csv'
                )

                hydroFilename = ''  # only the 'wetFilename' is used for the station_file

                # copy the input station file to the output location for editing
                indf = pd.read_csv(args.aoi._filename).drop_duplicates(subset=['Lat', 'Lon'])
                indf.to_csv(wetFilename, index=False)

            else:
                # This implies rasters
                fmt = get_raster_ext(args.file_format)
                wetFilename, hydroFilename = makeDelayFileNames(
                    d,
                    args.los,
                    fmt,
                    args.weather_model._dataset.upper(),
                    args.output_directory,
                )

        else:
            # In this case a cube file format is needed
            if args.file_format not in '.nc .h5 h5 hdf5 .hdf5 nc'.split():
                fmt = 'nc'
                logger.debug('Invalid extension %s for cube. Defaulting to .nc', args.file_format)
            else:
                fmt = args.file_format.strip('.').replace('df', '')

            wetFilename, hydroFilename = makeDelayFileNames(
                d,
                args.los,
                fmt,
                args.weather_model._dataset.upper(),
                args.output_directory,
            )

        wetNames.append(wetFilename)
        hydroNames.append(hydroFilename)

    args.wetFilenames = wetNames
    args.hydroFilenames = hydroNames

    return args


def get_raster_ext(fmt):
    drivers = rd.raster_driver_extensions()
    extensions = {value.upper(): key for key, value in drivers.items()}

    # add in ENVI/ISCE formats with generic extension
    extensions['ENVI'] = '.dat'
    extensions['ISCE'] = '.dat'

    try:
        return extensions[fmt.upper()]
    except KeyError:
        raise ValueError(f'{fmt} is not a valid gdal/rasterio file format for rasters')


def makeDelayFileNames(time: Optional[datetime], los: Optional[LOS], outformat: str, weather_model_name: str, out: Path) -> tuple[str, str]:
    """
    return names for the wet and hydrostatic delays.

    # Examples:
    >>> makeDelayFileNames(datetime(2020, 1, 1, 0, 0, 0), None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_00_00_00_ztd.h5', 'some_dir/model_name_hydro_00_00_00_ztd.h5')
    >>> makeDelayFileNames(None, None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_ztd.h5', 'some_dir/model_name_hydro_ztd.h5')
    """
    format_string = '{model_name}_{{}}_{time}{los}.{ext}'.format(
        model_name=weather_model_name,
        time=time.strftime('%Y%m%dT%H%M%S_') if time is not None else '',
        los='ztd' if (isinstance(los, Zenith) or los is None) else 'std',
        ext=outformat,
    )
    hydroname, wetname = (format_string.format(dtyp) for dtyp in ('hydro', 'wet'))

    hydro_file_name = str(out / hydroname)
    wet_file_name = str(out / wetname)
    return wet_file_name, hydro_file_name
