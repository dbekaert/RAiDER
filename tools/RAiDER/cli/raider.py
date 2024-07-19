import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path
from textwrap import dedent
from typing import Literal, Optional, cast

from RAiDER.types import CalcDelaysArgs, CalcDelaysArgsUnvalidated, TimeInterpolationMethod
import numpy as np
import xarray as xr
import yaml

import RAiDER.aria.calcGUNW
import RAiDER.aria.prepFromGUNW
from RAiDER import aws
from RAiDER.cli import DEFAULT_DICT, AttributeDict
from RAiDER.cli.parser import add_cpus, add_out, add_verbose
from RAiDER.cli.validators import DateListAction, date_type
from RAiDER.logger import logger, logging
from RAiDER.models.allowed import ALLOWED_MODELS
from RAiDER.models.customExceptions import DatetimeFailed, NoWeatherModelData, TryToKeepGoingError, WrongNumberOfFiles
from RAiDER.s1_azimuth_timing import (
    get_inverse_weights_for_dates,
    get_s1_azimuth_time_grid,
    get_times_for_azimuth_interpolation,
)
from RAiDER.utilFcns import get_dt


TIME_INTERPOLATION_METHODS = ['none', 'center_time', 'azimuth_time_grid']


HELP_MESSAGE = """
Command line options for RAiDER processing. Default options can be found by running
raider.py --generate_config

Download a weather model and calculate tropospheric delays
"""

EXAMPLES = """
Usage examples:
raider.py -g
raider.py run_config_file.yaml
"""

DEFAULT_RUN_CONFIG_PATH = os.path.abspath('./raider.yaml')


def read_run_config_file(fname):
    """
    Read the run config file into a dictionary structure.

    Args:
        fname (str): full path to the run config file
    Returns:
        dict: arguments to pass to RAiDER functions

    Examples:
    >>> run_config = read_run_config_file('raider.yaml')

    """
    from RAiDER.cli.validators import enforce_time, enforce_wm, get_heights, get_los, get_query_region, parse_dates

    with open(fname) as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError(f'Something is wrong with the yaml file {fname}')

    # Drop any values not specified
    params = drop_nans(params)

    # Need to ensure that all the groups exist, even if they are not specified by the user
    group_keys = ['date_group', 'time_group', 'aoi_group', 'height_group', 'los_group', 'runtime_group']
    for key in group_keys:
        if key not in params.keys():
            params[key] = {}

    # Parse the user-provided arguments
    run_config = DEFAULT_DICT.copy()
    for key, value in params.items():
        if key == 'runtime_group':
            for k, v in value.items():
                if v is not None:
                    run_config[k] = v
        if key == 'time_group':
            run_config.update(enforce_time(AttributeDict(value)))
        if key == 'date_group':
            run_config['date_list'] = parse_dates(AttributeDict(value))
        if key == 'aoi_group':
            # in case a DEM is passed and should be used
            dct_temp = {**AttributeDict(value), **AttributeDict(params['height_group'])}
            run_config['aoi'] = get_query_region(AttributeDict(dct_temp))

        if key == 'los_group':
            run_config['los'] = get_los(AttributeDict(value))
            run_config['zref'] = AttributeDict(value).get('zref')
        if key == 'look_dir':
            if value.lower() not in ['right', 'left']:
                raise ValueError(f'Unknown look direction {value}')
            run_config['look_dir'] = value.lower()
        if key == 'cube_spacing_in_m':
            run_config[key] = float(value) if isinstance(value, str) else value
        if key == 'download_only':
            run_config[key] = bool(value)

    # Have to guarantee that certain variables exist prior to looking at heights
    for key, value in params.items():
        if key == 'height_group':
            run_config.update(
                get_heights(
                    AttributeDict(value),
                    run_config['output_directory'],
                    run_config['station_file'],
                    run_config['bounding_box'],
                )
            )

        if key == 'weather_model':
            run_config[key] = enforce_wm(value, run_config['aoi'])

    run_config['aoi']._cube_spacing_m = run_config['cube_spacing_in_m']
    return AttributeDict(run_config)


def drop_nans(d):
    for key in list(d.keys()):
        if d[key] is None:
            del d[key]
        elif isinstance(d[key], dict):
            for k in list(d[key].keys()):
                if d[key][k] is None:
                    del d[key][k]
    return d


def calcDelays(iargs=None) -> list[Path]:
    """Parse command line arguments using argparse."""
    import RAiDER
    import RAiDER.processWM
    from RAiDER.checkArgs import checkArgs
    from RAiDER.delay import tropo_delay
    from RAiDER.utilFcns import get_nearest_wmtimes, writeDelays

    examples = 'Examples of use:' \
        '\n\t raider.py run_config_file.yaml' \
        '\n\t raider.py --generate_config template'

    p = argparse.ArgumentParser(
        description=HELP_MESSAGE,
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument(
        '--download_only',
        action='store_true',
        help='only download a weather model.'
    )

    # Generate an example configuration file, OR
    # run with a configuration file.
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--generate_config',
        '-g',
        nargs='?',
        choices=[
            'template',
            'example_LA_bbox',
            'example_LA_GNSS',
            'example_UK_isce',
        ],
        help='Generate an example run configuration and exit',
    )
    group.add_argument(
        'run_config_file',
        nargs='?',
        help='a YAML file with arguments to RAiDER'
    )

    # if not None, will replace first argument (run_config_file)
    args = p.parse_args(args=iargs)

    # Default example run configuration file
    ex_run_config_name = args.generate_config or 'template'
    ex_run_config_dir = Path(RAiDER.__file__).parent / 'cli/examples' / ex_run_config_name

    if args.generate_config is not None:
        for filename in ex_run_config_dir.glob('*'):
            dest_path = Path(os.getcwd()) / filename.name
            if dest_path.exists():
                print(f'File {dest_path} already exists. Overwrite? [y/n]')
                if input().lower() != 'y':
                    continue
            shutil.copy(filename, os.getcwd())
            logger.info('Wrote: %s', filename)
        sys.exit()
    # args.generate_config now guaranteed to be None

    # If no run configuration file is provided, look for a ./raider.yaml
    if args.run_config_file is not None:
        if not os.path.isfile(args.run_config_file):
            raise FileNotFoundError(args.run_config_file)
    else:
        if not os.path.isfile(DEFAULT_RUN_CONFIG_PATH):
            msg = (
                'No run configuration file provided! Specify a run configuration '
                "file or have a 'raider.yaml' file in the current directory."
            )
            p.print_usage()
            print(examples)
            raise SystemExit(f'ERROR: {msg}')
        args.run_config_file = DEFAULT_RUN_CONFIG_PATH

    # Read the run config file
    params = read_run_config_file(args.run_config_file)

    # Verify the run config file's parameters
    params = checkArgs(params)
    dl_only = params['download_only'] or args.download_only

    if not params.verbose:
        logger.setLevel(logging.INFO)

    # Extract and buffer the AOI
    los = params['los']
    aoi = params['aoi']
    model = params['weather_model']

    # adjust user requested AOI by grid size and buffer slightly
    aoi.add_buffer(model.getLLRes())

    # define the xy grid within the buffered bounding box
    aoi.set_output_xygrid(params['output_projection'])

    # add a buffer determined by latitude for ray tracing
    if los.ray_trace():
        wm_bounds = aoi.calc_buffer_ray(los.getSensorDirection(), lookDir=los.getLookDirection(), incAngle=30)
    else:
        wm_bounds = aoi.bounds()

    model.set_latlon_bounds(wm_bounds, output_spacing=aoi.get_output_spacing())

    wet_paths: list[Path] = []
    for t, w, f in zip(params['date_list'], params['wetFilenames'], params['hydroFilenames']):
        ###########################################################
        # Weather model calculation
        ###########################################################
        logger.debug('Starting to run the weather model calculation')
        logger.debug(f'Requested date,time: {t.strftime("%Y%m%d, %H:%M")}')
        logger.debug('Beginning weather model pre-processing')

        interp_method = params.get('interpolate_time')
        if interp_method is None:
            interp_method = 'none'
            logger.warning(
                "interp_method is not specified, defaulting to 'none', i.e. nearest datetime for delay calculation"
            )

        if interp_method != 'azimuth_time_grid':
            times = (
                get_nearest_wmtimes(t, [model.dtime() if model.dtime() is not None else 6][0])
                if interp_method == 'center_time'
                else [t]
            )

        elif interp_method == 'azimuth_time_grid':
            step = model.dtime()
            time_step_hours = step if step is not None else 6
            # Will yield 2 or 3 dates depending if t is within 5 minutes of time step
            times = get_times_for_azimuth_interpolation(t, time_step_hours)
        else:
            raise NotImplementedError(
                'Only none, center_time, and azimuth_time_grid are accepted values for interp_method.'
            )
        wfiles = []
        for tt in times:
            try:
                wfile = RAiDER.processWM.prepareWeatherModel(model, tt, aoi.bounds(), makePlots=params['verbose'])
                wfiles.append(wfile)

            except TryToKeepGoingError:
                if interp_method in ['azimuth_time_grid', 'none']:
                    raise DatetimeFailed(model.Model(), tt)
                else:
                    continue

            # log when something else happens and then continue with the next time
            except Exception as e:
                S, N, W, E = wm_bounds
                logger.info('Weather model point bounds are ' f'{S:.2f}/{N:.2f}/{W:.2f}/{E:.2f}')
                logger.info(f'Query datetime: {tt}')
                logger.error(e)
                logger.error(f'Weather model files are: {wfiles}')
                logger.error(f'Downloading and/or preparation of {model._Name} failed.')
                continue

        # dont process the delays for download only
        if dl_only:
            continue

        # Get the weather model file
        weather_model_file = getWeatherFile(wfiles, times, t, model._Name, interp_method)

        # Now process the delays
        try:
            wet_delay, hydro_delay = tropo_delay(
                t,
                weather_model_file,
                aoi,
                los,
                height_levels=params['height_levels'],
                out_proj=params['output_projection'],
                zref=params['zref'],
            )
        except RuntimeError:
            logger.exception('Datetime %s failed', t)
            continue

        # Different options depending on the inputs
        if los.is_Projected():
            out_filename = w.replace('_ztd', '_std')
            f = f.replace('_ztd', '_std')
        elif los.ray_trace():
            out_filename = w.replace('_std', '_ray')
            f = f.replace('_std', '_ray')
        else:
            out_filename = w

        # A dataset was returned by the above
        # Dataset returned: Cube e.g. GUNW workflow
        if hydro_delay is None:
            ds = wet_delay
            ext = os.path.splitext(out_filename)[1]
            out_filename = out_filename.replace('wet', 'tropo')

            # data provenance: include metadata for model and times used
            times_str = [t.strftime('%Y%m%dT%H:%M:%S') for t in sorted(times)]
            ds = ds.assign_attrs(model_name=model._Name, model_times_used=times_str, interpolation_method=interp_method)
            if ext not in ['.nc', '.h5']:
                out_filename = f'{os.path.splitext(out_filename)[0]}.nc'

            if out_filename.endswith('.nc'):
                ds.to_netcdf(out_filename, mode='w')
            elif out_filename.endswith('.h5'):
                ds.to_netcdf(out_filename, engine='h5netcdf', invalid_netcdf=True)

            logger.info('\nSuccessfully wrote delay cube to: %s\n', out_filename)
        # Dataset returned: station files, radar_raster, geocoded_file
        else:
            if aoi.type() == 'station_file':
                out_filename = f'{os.path.splitext(out_filename)[0]}.csv'

            if aoi.type() in ['station_file', 'radar_rasters', 'geocoded_file']:
                writeDelays(aoi, wet_delay, hydro_delay, out_filename, f, outformat=params['raster_format'])

        wet_paths.append(Path(out_filename))

    return wet_paths


# ------------------------------------------------------ downloadGNSSDelays.py
def downloadGNSS() -> None:
    """Parse command line arguments using argparse."""
    from RAiDER.gnss.downloadGNSSDelays import main as dlGNSS

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=""" \
    Check for and download tropospheric zenith delays for a set of GNSS stations from UNR

    Example call to virtually access and append zenith delay information to a CSV table in specified output
    directory, across specified range of time (in YYMMDD YYMMDD) and all available times of day, and confined to specified
    geographic bounding box :
    downloadGNSSdelay.py --out products -y 20100101 20141231 -b '39 40 -79 -78'

    Example call to virtually access and append zenith delay information to a CSV table in specified output
    directory, across specified range of time (in YYMMDD YYMMDD) and specified time of day, and distributed globally :
    downloadGNSSdelay.py --out products -y 20100101 20141231 --returntime '00:00:00'


    Example call to virtually access and append zenith delay information to a CSV table in specified output
    directory, across specified range of time in 12 day steps (in YYMMDD YYMMDD days) and specified time of day, and distributed globally :
    downloadGNSSdelay.py --out products -y 20100101 20141231 12 --returntime '00:00:00'

    Example call to virtually access and append zenith delay information to a CSV table in specified output
    directory, across specified range of time (in YYMMDD YYMMDD) and specified time of day, and distributed globally but restricted
    to list of stations specified in input textfile :
    downloadGNSSdelay.py --out products -y 20100101 20141231 --returntime '00:00:00' -f station_list.txt

    NOTE, following example call to physically download zenith delay information not recommended as it is not
    necessary for most applications.
    Example call to physically download and append zenith delay information to a CSV table in specified output
    directory, across specified range of time (in YYMMDD YYMMDD) and specified time of day, and confined to specified
    geographic bounding box :
    downloadGNSSdelay.py --download --out products -y 20100101 20141231 --returntime '00:00:00' -b '39 40 -79 -78'
    """,
    )

    # Stations to check/download
    area = p.add_argument_group(
        'Stations to check/download. Can be a lat/lon bounding box or file, or will run the whole world if not specified'
    )
    area.add_argument(
        '--station_file',
        '-f',
        default=None,
        dest='station_file',
        help=('Text file containing a list of 4-char station IDs separated by newlines'),
    )
    area.add_argument(
        '-b',
        '--bounding_box',
        dest='bounding_box',
        type=str,
        default=None,
        help="Provide either valid shapefile or Lat/Lon Bounding SNWE. -- Example : '19 20 -99.5 -98.5'",
    )
    area.add_argument(
        '--gpsrepo',
        '-gr',
        default='UNR',
        dest='gps_repo',
        help=('Specify GPS repository you wish to query. Currently supported archives: UNR.'),
    )

    misc = p.add_argument_group('Run parameters')
    add_out(misc)

    misc.add_argument(
        '--date',
        dest='dateList',
        help=dedent("""\
            Date to calculate delay.
            Can be a single date, a list of two dates (earlier, later) with 1-day interval, or a list of two dates and interval in days (earlier, later, interval).
            Example accepted formats:
               YYYYMMDD or
               YYYYMMDD YYYYMMDD
               YYYYMMDD YYYYMMDD N
        """),
        nargs="+",
        action=DateListAction,
        type=date_type,
        required=True,
    )

    misc.add_argument(
        '--returntime',
        dest='returnTime',
        help="Return delays closest to this specified time. If not specified, the GPS delays for all times will be returned. Input in 'HH:MM:SS', e.g. '16:00:00'",
        default=None,
    )

    misc.add_argument(
        '--download',
        help='Physically download data. Note this option is not necessary to proceed with statistical analyses, as data can be handled virtually in the program.',
        action='store_true',
        dest='download',
        default=False,
    )

    add_cpus(misc)
    add_verbose(misc)

    args = p.parse_args()

    dlGNSS(args)


# ------------------------------------------------------------ prepFromGUNW.py
def calcDelaysGUNW(iargs: Optional[list[str]] = None) -> xr.Dataset:
    p = argparse.ArgumentParser(
        description='Calculate a cube of interferometic delays for GUNW files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        '--bucket',
        help='S3 bucket containing ARIA GUNW NetCDF file. Will be ignored if the --file argument is provided.',
    )

    p.add_argument(
        '--bucket-prefix',
        default='',
        help='S3 bucket prefix which may contain an ARIA GUNW NetCDF file to calculate delays for and which the final '
        'ARIA GUNW NetCDF file will be upload to. Will be ignored if the --file argument is provided.',
    )

    p.add_argument(
        '--input-bucket-prefix',
        help='S3 bucket prefix that contains an ARIA GUNW NetCDF file to calculate delays for. '
        'If not provided, will look in --bucket-prefix for an ARIA GUNW NetCDF file. '
        'Will be ignored if the --file argument is provided.',
    )

    p.add_argument(
        '-f',
        '--file',
        help='1 ARIA GUNW netcdf file'
    )

    p.add_argument(
        '-m',
        '--weather-model',
        default='HRRR',
        choices=['None'] + ALLOWED_MODELS,
        help='Weather model.'
    )

    p.add_argument(
        '-uid',
        '--api_uid',
        default=None,
        help='Weather model API UID [uid, email, username], depending on model.',
    )

    p.add_argument(
        '-key',
        '--api_key',
        default=None,
        help='Weather model API KEY [key, password], depending on model.'
    )

    p.add_argument(
        '-interp',
        '--interpolate-time',
        default='azimuth_time_grid',
        choices=TIME_INTERPOLATION_METHODS,
        help=(
            'How to interpolate across model time steps. Possible options are: '
            "['none', 'center_time', 'azimuth_time_grid'] "
            'None: means nearest model time; center_time: linearly across center time; '
            'Azimuth_time_grid: means every pixel is weighted with respect to azimuth time of S1'
        ),
    )

    p.add_argument(
        '-o',
        '--output-directory',
        default=str(Path.cwd()),
        help='Directory to store results.'
    )

    args: CalcDelaysArgsUnvalidated = p.parse_args(iargs, namespace=CalcDelaysArgsUnvalidated())

    if args.input_bucket_prefix is None:
        args.input_bucket_prefix = args.bucket_prefix

    if args.interpolate_time not in TIME_INTERPOLATION_METHODS:
        raise ValueError("interpolate_time arg must be in ['none', 'center_time', 'azimuth_time_grid']")
    args.interpolate_time = cast(TimeInterpolationMethod, args.interpolate_time)

    if args.weather_model == 'None':
        # NOTE: HyP3's current step function implementation does not have a good way of conditionally
        #       running processing steps. This allows HyP3 to always run this step but exit immediately
        #       and do nothing if tropospheric correction via RAiDER is not selected. This should not cause
        #       any appreciable cost increase to GUNW product generation.
        print('Nothing to do!')
        return

    if (
        args.file is not None and
        args.weather_model == 'HRRR' and
        args.interpolate_time == 'azimuth_time_grid'
    ):
        file_name = args.file.split('/')[-1]
        gunw_id = file_name.replace('.nc', '')
        if not RAiDER.aria.prepFromGUNW.check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation(gunw_id):
            raise NoWeatherModelData('The required HRRR data for time-grid interpolation is not available')

    if args.file is None:
        if args.bucket is None:
            raise ValueError('Either argument --file or --bucket must be provided')

        # only use GUNW ID for checking if HRRR available
        args.file = aws.get_s3_file(args.bucket, args.input_bucket_prefix, '.nc')
        if args.file is None:
            raise ValueError(
                'GUNW product file could not be found at' f's3://{args.bucket}/{args.input_bucket_prefix}'
            )
        if args.weather_model == 'HRRR' and args.interpolate_time == 'azimuth_time_grid':
            gunw_nc_name = args.file.split('/')[-1]
            gunw_id = gunw_nc_name.replace('.nc', '')
            if not RAiDER.aria.prepFromGUNW.check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation(gunw_id):
                print(
                    'The required HRRR data for time-grid interpolation is not available; returning None and not modifying GUNW dataset'
                )
                return

        # Download file to obtain metadata
        if not RAiDER.aria.prepFromGUNW.check_weather_model_availability(args.file, args.weather_model):
            # NOTE: We want to submit jobs that are outside of acceptable weather model range
            #       and still deliver these products to the DAAC without this layer. Therefore
            #       we include this within this portion of the control flow.
            print('Nothing to do because outside of weather model range')
            return
        json_file_path = aws.get_s3_file(args.bucket, args.input_bucket_prefix, '.json')
        if json_file_path is None:
            raise ValueError(
                'GUNW metadata file could not be found at' f's3://{args.bucket}/{args.input_bucket_prefix}'
            )
        json_data = json.load(open(json_file_path))
        json_data['metadata'].setdefault('weather_model', []).append(args.weather_model)
        json.dump(json_data, open(json_file_path, 'w'))

        # also get browse image -- if RAiDER is running in its own HyP3 job, the browse image will be needed for ingest
        browse_file_path = aws.get_s3_file(args.bucket, args.input_bucket_prefix, '.png')
        if browse_file_path is None:
            raise ValueError(
                'GUNW browse image could not be found at' f's3://{args.bucket}/{args.input_bucket_prefix}'
            )

    args = cast(CalcDelaysArgs, args)

    # prep the config needed for delay calcs
    path_cfg, wavelength = RAiDER.aria.prepFromGUNW.main(args)

    # write delay cube (nc) to disk using config
    # return a list with the path to cube for each date
    cube_filenames = calcDelays([path_cfg])

    assert len(cube_filenames) == 2, 'Incorrect number of delay files written.'

    # calculate the interferometric phase and write it out
    ds = RAiDER.aria.calcGUNW.tropo_gunw_slc(
        cube_filenames,
        args.file,
        wavelength,
    )

    # upload to s3
    if args.bucket is not None:
        aws.upload_file_to_s3(args.file, args.bucket, args.bucket_prefix)
        aws.upload_file_to_s3(json_file_path, args.bucket, args.bucket_prefix)
        aws.upload_file_to_s3(browse_file_path, args.bucket, args.bucket_prefix)
    return ds


# ------------------------------------------------------------ processDelays.py
def combineZTDFiles() -> None:
    """Command-line program to process delay files from RAiDER and GNSS into a single file."""
    from RAiDER.gnss.processDelayFiles import combineDelayFiles, create_parser, main

    p = create_parser()
    args = p.parse_args()

    if not os.path.exists(args.raider_file):
        combineDelayFiles(args.raider_file, loc=args.raider_folder)

    if not os.path.exists(args.gnss_file):
        combineDelayFiles(
            args.gnss_file, loc=args.gnss_folder, source='GNSS', ref=args.raider_file, col_name=args.column_name
        )

    if args.gnss_file is not None:
        main(
            args.raider_file,
            args.gnss_file,
            col_name=args.column_name,
            raider_delay=args.raider_column_name,
            outName=args.out_name,
            localTime=args.local_time,
        )


def getWeatherFile(wfiles, times, t, model, interp_method='none'):
    """
    # Time interpolation.
    #
    # Need to handle various cases, including if the exact weather model time is
    # requested, or if one or more datetimes are not available from the weather
    # model data provider
    """
    # time interpolation method: number of expected files
    EXPECTED_NUM_FILES = {'none': 1, 'center_time': 2, 'azimuth_time_grid': 3}

    Nfiles = len(wfiles)
    Ntimes = len(times)

    try:
        Nfiles_expected = EXPECTED_NUM_FILES[interp_method]
    except KeyError:
        raise ValueError(f'getWeatherFile: interp_method {interp_method} is not known')

    Nmatch = Nfiles_expected == Nfiles
    Tmatch = Nfiles == Ntimes

    # Case 1: no files downloaded
    if Nfiles == 0:
        logger.error('No weather model data was successfully processed.')
        return None

    # Case 2 - nearest weather model time is requested and retrieved
    if interp_method == 'none':
        weather_model_file = wfiles[0]

    elif interp_method == 'center_time':
        if Nmatch:  # Case 3: two weather files downloaded
            weather_model_file = combine_weather_files(wfiles, t, model, interp_method='center_time')
        elif Tmatch:  # Case 4: Exact time is available without interpolation
            logger.warning('Time interpolation is not needed as exact time is available')
            weather_model_file = wfiles[0]
        elif Nfiles == 1:  # Case 5: one file does not download for some reason
            logger.warning(
                'getWeatherFile: One datetime is not available to download, defaulting to nearest available date'
            )
            weather_model_file = wfiles[0]
        else:
            raise WrongNumberOfFiles(Nfiles_expected, Nfiles)

    elif (interp_method) == 'azimuth_time_grid':
        if Nmatch or Tmatch:  # Case 6: all files downloaded
            weather_model_file = combine_weather_files(wfiles, t, model, interp_method='azimuth_time_grid')
        else:
            raise WrongNumberOfFiles(Nfiles_expected, Nfiles)

    # Case 7 - Anything else errors out
    else:
        N = len(wfiles)
        raise NotImplementedError(
            f'The {interp_method} with {N} retrieved weather model files was not well posed '
            'for the current workflow.'
        )

    return weather_model_file


def combine_weather_files(wfiles, t, model, interp_method='center_time'):
    """Interpolate downloaded weather files and save to a single file."""
    STYLE = {'center_time': '_timeInterp_', 'azimuth_time_grid': '_timeInterpAziGrid_'}

    # read the individual datetime datasets
    datasets = [xr.open_dataset(f) for f in wfiles]

    # Pull the datetimes from the datasets
    times = []
    for ds in datasets:
        times.append(datetime.datetime.strptime(ds.attrs['datetime'], '%Y_%m_%dT%H_%M_%S'))

    if len(times) == 0:
        raise NoWeatherModelData()

    # calculate relative weights of each dataset
    if interp_method == 'center_time':
        wgts = get_weights_time_interp(times, t)
    elif interp_method == 'azimuth_time_grid':
        time_grid = get_time_grid_for_aztime_interp(datasets, t, model)
        wgts = get_inverse_weights_for_dates(time_grid, times)

    # combine datasets
    ds_out = datasets[0]
    for var in ['wet', 'hydro', 'wet_total', 'hydro_total']:
        ds_out[var] = sum([wgt * ds[var] for (wgt, ds) in zip(wgts, datasets)])
    ds_out.attrs['Date1'] = 0
    ds_out.attrs['Date2'] = 0

    # Give the weighted combination a new file name
    weather_model_file = os.path.join(
        os.path.dirname(wfiles[0]),
        os.path.basename(wfiles[0]).split('_')[0]
        + '_'
        + t.strftime('%Y_%m_%dT%H_%M_%S')
        + STYLE[interp_method]
        + '_'.join(wfiles[0].split('_')[-4:]),
    )

    # write the combined results to disk
    ds_out.to_netcdf(weather_model_file)

    return weather_model_file


def combine_files_using_azimuth_time(wfiles, t, times):
    """Combine files using azimuth time interpolation."""
    # read the individual datetime datasets
    datasets = [xr.open_dataset(f) for f in wfiles]

    # Pull the datetimes from the datasets
    times = []
    for ds in datasets:
        times.append(datetime.datetime.strptime(ds.attrs['datetime'], '%Y_%m_%dT%H_%M_%S'))

    model = datasets[0].attrs['model_name']

    time_grid = get_time_grid_for_aztime_interp(datasets, times, t, model)

    wgts = get_inverse_weights_for_dates(time_grid, times)

    # combine datasets
    ds_out = datasets[0]
    for var in ['wet', 'hydro', 'wet_total', 'hydro_total']:
        ds_out[var] = sum([wgt * ds[var] for (wgt, ds) in zip(wgts, datasets)])
    ds_out.attrs['Date1'] = 0
    ds_out.attrs['Date2'] = 0

    # Give the weighted combination a new file name
    weather_model_file = os.path.join(
        os.path.dirname(wfiles[0]),
        os.path.basename(wfiles[0]).split('_')[0]
        + '_'
        + t.strftime('%Y_%m_%dT%H_%M_%S')
        + '_timeInterpAziGrid_'
        + '_'.join(wfiles[0].split('_')[-4:]),
    )

    # write the combined results to disk
    ds_out.to_netcdf(weather_model_file)

    return weather_model_file


def get_weights_time_interp(times, t):
    """Calculate weights for time interpolation using simple inverse linear weighting."""
    date1, date2 = times
    wgts = [1 - get_dt(t, date1) / get_dt(date2, date1), 1 - get_dt(date2, t) / get_dt(date2, date1)]

    try:
        assert np.isclose(np.sum(wgts), 1)
    except AssertionError:
        logger.error('Time interpolation weights do not sum to one; something is off with query datetime: %s', t)
        return None

    return wgts


def get_time_grid_for_aztime_interp(datasets, t, model):
    """Calculate the time-varying grid for use with azimuth time interpolation."""
    # Each model will require some inspection here
    # the subsequent s1 azimuth time grid requires dimension
    # inputs to all have same dimensions and either be
    # 1d or 3d.
    AZ_TIME_ALLOWED_MODELS = 'hrrr hrrrak hrrr-ak'.split()

    if model.lower() in AZ_TIME_ALLOWED_MODELS:
        lat_2d = datasets[0].latitude.data
        lon_2d = datasets[0].longitude.data
        z_1d = datasets[0].z.data
        m, n, p = z_1d.shape[0], lat_2d.shape[0], lat_2d.shape[1]

        lat = np.broadcast_to(lat_2d, (m, n, p))
        lon = np.broadcast_to(lon_2d, (m, n, p))
        hgt = np.broadcast_to(z_1d[:, None, None], (m, n, p))

    else:
        raise NotImplementedError('Azimuth Time is currently only implemented for HRRR')

    time_grid = get_s1_azimuth_time_grid(lon, lat, hgt, t)  # This is the acq time from loop
    if np.any(np.isnan(time_grid)):
        raise ValueError('The Time Grid return nans meaning no orbit was downloaded.')

    return time_grid
