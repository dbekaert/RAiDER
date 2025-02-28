# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd
import rasterio.drivers as rd

from RAiDER.cli.types import RunConfig
from RAiDER.llreader import BoundingBox, StationFile
from RAiDER.logger import logger
from RAiDER.losreader import LOS, Zenith


def checkArgs(run_config: RunConfig) -> RunConfig:
    """Check argument compatibility and return the correct variables."""
    ############################################################################
    # Directories
    run_config.runtime_group.output_directory.mkdir(exist_ok=True)
    run_config.runtime_group.weather_model_directory.mkdir(exist_ok=True)
    run_config.weather_model.set_wmLoc(str(run_config.runtime_group.weather_model_directory))

    ############################################################################
    # Date and Time parsing
    run_config.date_group.date_list = [
        dt.datetime.combine(d, run_config.time_group.time)
        for d in run_config.date_group.date_list
    ]
    if len(run_config.date_group.date_list) > 1 and run_config.los_group.orbit_file is not None:
        logger.warning(
            'Only one orbit file is being used to get the look vectors for all requested times. If you want to use '
            'separate orbit files you will need to run RAiDER separately for each time.'
        )

    run_config.los_group.los.setTime(run_config.date_group.date_list[0])

    ############################################################################
    # filenames
    wetNames: list[str] = []
    hydroNames: list[str] = []
    for d in run_config.date_group.date_list:
        if not isinstance(run_config.aoi_group.aoi, BoundingBox):
            # Handle the GNSS station file
            if isinstance(run_config.aoi_group.aoi, StationFile):
                wetFilename = str(
                    run_config.runtime_group.output_directory /
                    f'{run_config.weather_model._dataset.upper()}_Delay_{d.strftime("%Y%m%dT%H%M%S")}_ztd.csv'
                )

                hydroFilename = ''  # only the 'wetFilename' is used for the station_file

                # copy the input station file to the output location for editing
                indf = pd.read_csv(run_config.aoi_group.aoi._filename) \
                    .drop_duplicates(subset=['Lat', 'Lon'])
                indf.to_csv(wetFilename, index=False)

            else:
                # This implies rasters
                fmt = get_raster_ext(run_config.runtime_group.file_format)
                wetFilename, hydroFilename = makeDelayFileNames(
                    d,
                    run_config.los_group.los,
                    fmt,
                    run_config.weather_model._dataset.upper(),
                    run_config.runtime_group.output_directory,
                )

        else:
            # In this case a cube file format is needed
            if run_config.runtime_group.file_format not in '.nc .h5 h5 hdf5 .hdf5 nc'.split():
                fmt = 'nc'
                logger.debug('Invalid extension %s for cube. Defaulting to .nc', run_config.runtime_group.file_format)
            else:
                fmt = run_config.runtime_group.file_format.strip('.').replace('df', '')

            wetFilename, hydroFilename = makeDelayFileNames(
                d,
                run_config.los_group.los,
                fmt,
                run_config.weather_model._dataset.upper(),
                run_config.runtime_group.output_directory,
            )

        wetNames.append(wetFilename)
        hydroNames.append(hydroFilename)

    run_config.wetFilenames = wetNames
    run_config.hydroFilenames = hydroNames

    return run_config


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


def makeDelayFileNames(date: Optional[dt.date], los: Optional[LOS], outformat: str, weather_model_name: str, out: Path) -> tuple[str, str]:
    """
    return names for the wet and hydrostatic delays.

    # Examples:
    >>> makeDelayFileNames(dt.datetime(2020, 1, 1, 0, 0, 0), None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_00_00_00_ztd.h5', 'some_dir/model_name_hydro_00_00_00_ztd.h5')
    >>> makeDelayFileNames(None, None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_ztd.h5', 'some_dir/model_name_hydro_ztd.h5')
    """
    format_string = '{model_name}_{{}}_{time}{los}.{ext}'.format(
        model_name=weather_model_name,
        time=date.strftime('%Y%m%dT%H%M%S_') if date is not None else '',
        los='ztd' if (isinstance(los, Zenith) or los is None) else 'std',
        ext=outformat,
    )
    hydroname, wetname = (format_string.format(dtyp) for dtyp in ('hydro', 'wet'))

    hydro_file_name = str(out / hydroname)
    wet_file_name = str(out / wetname)
    return wet_file_name, hydro_file_name
