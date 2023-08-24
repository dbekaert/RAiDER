import argparse
import datetime
import os
import json
import shutil
import sys
import yaml

import numpy as np
import xarray as xr

from textwrap import dedent

import RAiDER.aria.prepFromGUNW
import RAiDER.aria.calcGUNW
from RAiDER import aws
from RAiDER.constants import _ZREF
from RAiDER.logger import logger, logging
from RAiDER.cli import DEFAULT_DICT, AttributeDict
from RAiDER.cli.parser import add_out, add_cpus, add_verbose
from RAiDER.cli.validators import DateListAction, date_type
from RAiDER.models.allowed import ALLOWED_MODELS
from RAiDER.utilFcns import get_dt
from RAiDER.s1_azimuth_timing import get_s1_azimuth_time_grid, get_inverse_weights_for_dates, get_times_for_azimuth_interpolation


import traceback


HELP_MESSAGE = """
Command line options for RAiDER processing. Default options can be found by running
raider.py --generate_config

Download a weather model and calculate tropospheric delays
"""

SHORT_MESSAGE = """
Program to calculate troposphere total delays using a weather model
"""

EXAMPLES = """
Usage examples:
raider.py -g
raider.py customTemplatefile.cfg
"""


def read_template_file(fname):
    """
    Read the template file into a dictionary structure.
    Args:
        fname (str): full path to the template file
    Returns:
        dict: arguments to pass to RAiDER functions

    Examples:
    >>> template = read_template_file('raider.yaml')

    """
    from RAiDER.cli.validators import (
        enforce_time, parse_dates, get_query_region, get_heights, get_los, enforce_wm
    )
    with open(fname, 'r') as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError('Something is wrong with the yaml file {}'.format(fname))

    # Drop any values not specified
    params = drop_nans(params)

    # Need to ensure that all the groups exist, even if they are not specified by the user
    group_keys = ['date_group', 'time_group', 'aoi_group', 'height_group', 'los_group', 'runtime_group']
    for key in group_keys:
        if not key in params.keys():
            params[key] = {}

    # Parse the user-provided arguments
    template = DEFAULT_DICT.copy()
    for key, value in params.items():
        if key == 'runtime_group':
            for k, v in value.items():
                if v is not None:
                    template[k] = v
        if key == 'time_group':
            template.update(enforce_time(AttributeDict(value)))
        if key == 'date_group':
            template['date_list'] = parse_dates(AttributeDict(value))
        if key == 'aoi_group':
            ## in case a DEM is passed and should be used
            dct_temp = {**AttributeDict(value),
                        **AttributeDict(params['height_group'])}
            template['aoi'] = get_query_region(AttributeDict(dct_temp))

        if key == 'los_group':
            template['los']  = get_los(AttributeDict(value))
            template['zref'] = AttributeDict(value).get('zref')
        if key == 'look_dir':
            if value.lower() not in ['right', 'left']:
                raise ValueError(f"Unknown look direction {value}")
            template['look_dir'] = value.lower()
        if key == 'cube_spacing_in_m':
            template[key] = float(value) if isinstance(value, str) else value
        if key == 'download_only':
            template[key] = bool(value)

    # Have to guarantee that certain variables exist prior to looking at heights
    for key, value in params.items():
        if key == 'height_group':
            template.update(
                get_heights(
                    AttributeDict(value),
                    template['output_directory'],
                    template['station_file'],
                    template['bounding_box'],
                )
            )

        if key == 'weather_model':
            template[key]= enforce_wm(value, template['aoi'])

    template['aoi']._cube_spacing_m = template['cube_spacing_in_m']
    return AttributeDict(template)


def drop_nans(d):
    for key in list(d.keys()):
        if d[key] is None:
            del d[key]
        elif isinstance(d[key], dict):
            for k in list(d[key].keys()):
                if d[key][k] is None:
                    del d[key][k]
    return d


def calcDelays(iargs=None):
    """ Parse command line arguments using argparse. """
    import RAiDER
    import RAiDER.processWM
    from RAiDER.delay import tropo_delay
    from RAiDER.checkArgs import checkArgs
    from RAiDER.utilFcns import writeDelays, get_nearest_wmtimes
    examples = 'Examples of use:' \
        '\n\t raider.py customTemplatefile.cfg' \
        '\n\t raider.py -g'

    p = argparse.ArgumentParser(
        description =
            'Command line interface for RAiDER processing with a configure file.'
            'Default options can be found by running: raider.py --generate_config',
        epilog=examples, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument(
        'customTemplateFile', nargs='?',
        help='custom template with option settings.\n' +
        "ignored if the default smallbaselineApp.cfg is input."
    )

    p.add_argument(
        '-g', '--generate_template', action='store_true',
        help='generate default template (if it does not exist) and exit.'
    )

    p.add_argument(
        '--download_only', action='store_true',
        help='only download a weather model.'
    )

    ## if not None, will replace first argument (customTemplateFile)
    args = p.parse_args(args=iargs)

    # default input file
    template_file = os.path.join(os.path.dirname(RAiDER.__file__),
                                                        'cli', 'raider.yaml')

    if args.generate_template:
        dst = os.path.join(os.getcwd(), 'raider.yaml')
        shutil.copyfile(template_file, dst)
        logger.info('Wrote: %s', dst)
        sys.exit()


    # check: existence of input template files
    if (not args.customTemplateFile
            and not os.path.isfile(os.path.basename(template_file))
            and not args.generate_template):
        msg = "No template file found! It requires that either:"
        msg += "\n  a custom template file, OR the default template "
        msg += "\n  file 'raider.yaml' exists in current directory."

        p.print_usage()
        print(examples)
        raise SystemExit(f'ERROR: {msg}')

    if  args.customTemplateFile:
        # check the existence
        if not os.path.isfile(args.customTemplateFile):
            raise FileNotFoundError(args.customTemplateFile)

        args.customTemplateFile = os.path.abspath(args.customTemplateFile)
    else:
        args.customTemplateFile = template_file

    # Read the template file
    params = read_template_file(args.customTemplateFile)

    # Argument checking
    params  = checkArgs(params)
    dl_only = True if params['download_only'] or args.download_only else False

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
        wm_bounds = aoi.calc_buffer_ray(los.getSensorDirection(),
                                lookDir=los.getLookDirection(), incAngle=30)
    else:
        wm_bounds = aoi.bounds()

    model.set_latlon_bounds(wm_bounds, output_spacing=aoi.get_output_spacing())

    wet_filenames = []
    for t, w, f in zip(
        params['date_list'],
        params['wetFilenames'],
        params['hydroFilenames']
    ):

        ###########################################################
        # weather model calculation
        logger.debug('Starting to run the weather model calculation')
        logger.debug(f'Requested date,time: {t.strftime("%Y%m%d, %H:%M")}')
        logger.debug('Beginning weather model pre-processing')

        interp_method = params.get('interpolate_time')
        if interp_method is None:
            interp_method = 'none'
            logger.warning('interp_method is not specified, defaulting to \'none\', i.e. nearest datetime for delay '
                           'calculation')

        # Grab the closest two times unless the user specifies 'nearest' via 'none' or None.
        # If the model time_delta is not specified then use 6
        # The two datetimes will be combined to a single file and processed
        # TODO: make more transparent control flow for GUNW and non-GUNW workflow
        if (interp_method in ['none', 'center_time']):
            times = get_nearest_wmtimes(t, [model.dtime() if \
                                        model.dtime() is not None else 6][0]) if interp_method == 'center_time' else [t]
        elif interp_method == 'azimuth_time_grid':
            step = model.dtime()
            time_step_hours = step if step is not None else 6
            # Will yield 2 or 3 dates depending if t is within 5 minutes of time step
            times = get_times_for_azimuth_interpolation(t, time_step_hours)
        else:
            raise NotImplementedError('Only none, center_time, and azimuth_time_grid are accepted values for '
                                      'interp_method.')
        wfiles = []
        for tt in times:
            try:
                wfile = RAiDER.processWM.prepareWeatherModel(model,
                                                             tt,
                                                             aoi.bounds(),
                                                             makePlots=params['verbose'])
                wfiles.append(wfile)

            # catch when requested datetime fails
            except RuntimeError as re:
                continue

            # catch when something else within weather model class fails
            except Exception as e:
                S, N, W, E = wm_bounds
                logger.info(f'Weather model point bounds are {S:.2f}/{N:.2f}/{W:.2f}/{E:.2f}')
                logger.info(f'Query datetime: {tt}')
                msg = f'Downloading and/or preparation of {model._Name} failed.'
                logger.error(e)
                logger.error(msg)
                if interp_method == 'azimuth_time_grid':
                    break
        # dont process the delays for download only
        if dl_only:
            continue

        if (len(wfiles) == 0) and (interp_method != 'azimuth_time_grid'):
            logger.error('No weather model data was successfully processed.')
            if len(params['date_list']) == 1:
                raise RuntimeError
            # skip date and continue processing if multiple dates are requested
            else:
                continue

        # nearest weather model time via 'none' is specified
        # When interp_method is 'none' only 1 weather model file and one relevant time
        elif (interp_method == 'none') and (len(wfiles) == 1) and (len(times) == 1):
            weather_model_file = wfiles[0]

        # only one time in temporal interpolation worked
        # TODO: this seems problematic - unexpected behavior possibly for 'center_time'
        elif (len(wfiles) == 1) and (len(times) == 2) and (interp_method != 'azimuth_time_grid'):
            logger.warning('Time interpolation did not succeed, defaulting to nearest available date')
            weather_model_file = wfiles[0]
        
        elif (interp_method == 'center_time')  and len(times)==1:
            logger.info('Requested time is provided exactly, will use only one weather model datetime')
            weather_model_file = wfiles[0]

        # TODO: ensure this additional conditional is appropriate; assuming wfiles == 2 ONLY for 'center_time'
        #  value of 'interp_method' parameter
        elif (interp_method == 'center_time') and (len(wfiles) == 2):
            ds1 = xr.open_dataset(wfiles[0])
            ds2 = xr.open_dataset(wfiles[1])

            # calculate relative weights of each dataset
            date1 = datetime.datetime.strptime(ds1.attrs['datetime'], '%Y_%m_%dT%H_%M_%S')
            date2 = datetime.datetime.strptime(ds2.attrs['datetime'], '%Y_%m_%dT%H_%M_%S')
            wgts  = [ 1 - get_dt(t, date1) / get_dt(date2, date1), 1 - get_dt(date2, t) / get_dt(date2, date1)]
            try:
                assert np.isclose(np.sum(wgts), 1)
            except AssertionError:
                logger.error('Time interpolation weights do not sum to one; something is off with query datetime: %s', t)
                continue

            # combine datasets
            ds = ds1
            for var in ['wet', 'hydro', 'wet_total', 'hydro_total']:
                ds[var] = (wgts[0] * ds1[var]) + (wgts[1] * ds2[var])
            ds.attrs['Date1'] = 0
            ds.attrs['Date2'] = 0
            weather_model_file = os.path.join(
                os.path.dirname(wfiles[0]),
                os.path.basename(wfiles[0]).split('_')[0] + '_' + t.strftime('%Y_%m_%dT%H_%M_%S') + '_timeInterp_' + '_'.join(wfiles[0].split('_')[-4:]),
            )
            ds.to_netcdf(weather_model_file)
        elif (interp_method == 'azimuth_time_grid'):
            n_files = len(wfiles)
            n_times = len(times)
            if n_files != n_times:
                raise ValueError('The model files for the datetimes for requisite azimuth interpolation were not '
                                 'succesfully downloaded or processed')
            datasets = [xr.open_dataset(f) for f in wfiles]

            # Each model will require some inspection here
            # the subsequent s1 azimuth time grid requires dimension
            # inputs to all have same dimensions and either be
            # 1d or 3d.
            if model._dataset in ['hrrr', 'hrrrak']:
                lat_2d = datasets[0].latitude.data
                lon_2d = datasets[0].longitude.data
                z_1d = datasets[0].z.data
                m, n, p = z_1d.shape[0], lat_2d.shape[0], lat_2d.shape[1]

                lat = np.broadcast_to(lat_2d, (m, n, p))
                lon = np.broadcast_to(lon_2d, (m, n, p))
                hgt = np.broadcast_to(z_1d[:, None, None], (m, n, p))

            else:
                raise NotImplementedError('Azimuth Time is currently only implemented for HRRR')

            time_grid = RAiDER.s1_azimuth_timing.get_s1_azimuth_time_grid(lon,
                                                                          lat,
                                                                          hgt,
                                                                          # This is the acq time from loop
                                                                          t)

            if np.any(np.isnan(time_grid)):
                raise ValueError('The azimuth time grid return nans meaning no orbit was downloaded.')
            wgts = get_inverse_weights_for_dates(time_grid,
                                                 times,
                                                 temporal_window_hours=model._time_res)
            # combine datasets
            ds_out = datasets[0]
            for var in ['wet', 'hydro', 'wet_total', 'hydro_total']:
                ds_out[var] = sum([wgt * ds[var] for (wgt, ds) in zip(wgts, datasets)])
            ds_out.attrs['Date1'] = 0
            ds_out.attrs['Date2'] = 0
            weather_model_file = os.path.join(
                os.path.dirname(wfiles[0]),
                # TODO: clean up
                os.path.basename(wfiles[0]).split('_')[0] + '_' + t.strftime('%Y_%m_%dT%H_%M_%S') + '_timeInterpAziGrid_' + '_'.join(wfiles[0].split('_')[-4:]),
            )
            ds_out.to_netcdf(weather_model_file)

        # TODO: test to ensure this error is caught
        else:
            n = len(wfiles)
            raise NotImplementedError(f'The {interp_method} with {n} retrieved weather model files was not well posed '
                                      'for the the delay workflow.')

        # Now process the delays
        try:
            wet_delay, hydro_delay = tropo_delay(
                t, weather_model_file, aoi, los,
                height_levels = params['height_levels'],
                out_proj = params['output_projection'],
                zref=params['zref']
            )
        except RuntimeError:
            logger.exception("Datetime %s failed", t)
            continue

        # Different options depending on the inputs
        if los.is_Projected():
            out_filename = w.replace("_ztd", "_std")
            f = f.replace("_ztd", "_std")
        elif los.ray_trace():
            out_filename = w.replace("_std", "_ray")
            f = f.replace("_std", "_ray")
        else:
            out_filename = w

        # A dataset was returned by the above
        # Dataset returned: Cube e.g. GUNW workflow
        if hydro_delay is None:
            ds = wet_delay
            ext = os.path.splitext(out_filename)[1]
            out_filename = out_filename.replace('wet', 'tropo')

            # data provenance: include metadata for model and times used
            times_str = [t.strftime("%Y%m%dT%H:%M:%S") for t in sorted(times)]
            ds = ds.assign_attrs(model_name=model._Name,
                                 model_times_used=times_str,
                                 interpolation_method=interp_method)
            if ext not in ['.nc', '.h5']:
                out_filename = f'{os.path.splitext(out_filename)[0]}.nc'

            if out_filename.endswith(".nc"):
                ds.to_netcdf(out_filename, mode="w")
            elif out_filename.endswith(".h5"):
                ds.to_netcdf(out_filename, engine="h5netcdf", invalid_netcdf=True)

            logger.info('\nSuccessfully wrote delay cube to: %s\n', out_filename)
        # Dataset returned: station files, radar_raster, geocoded_file
        else:
            if aoi.type() == 'station_file':
                out_filename = f'{os.path.splitext(out_filename)[0]}.csv'

            if aoi.type() in ['station_file', 'radar_rasters', 'geocoded_file']:
                writeDelays(aoi, wet_delay, hydro_delay, out_filename, f, outformat=params['raster_format'])

        wet_filenames.append(out_filename)

    return wet_filenames


## ------------------------------------------------------ downloadGNSSDelays.py
def downloadGNSS():
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
    """)

    # Stations to check/download
    area = p.add_argument_group(
        'Stations to check/download. Can be a lat/lon bounding box or file, or will run the whole world if not specified')
    area.add_argument(
        '--station_file', '-f', default=None, dest='station_file',
        help=('Text file containing a list of 4-char station IDs separated by newlines'))
    area.add_argument(
        '-b', '--bounding_box', dest='bounding_box', type=str, default=None,
        help="Provide either valid shapefile or Lat/Lon Bounding SNWE. -- Example : '19 20 -99.5 -98.5'")
    area.add_argument(
        '--gpsrepo', '-gr', default='UNR', dest='gps_repo',
        help=('Specify GPS repository you wish to query. Currently supported archives: UNR.'))

    misc = p.add_argument_group("Run parameters")
    add_out(misc)

    misc.add_argument(
        '--date', dest='dateList',
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
        required=True
    )

    misc.add_argument(
        '--returntime', dest='returnTime',
        help="Return delays closest to this specified time. If not specified, the GPS delays for all times will be returned. Input in 'HH:MM:SS', e.g. '16:00:00'",
        default=None)

    misc.add_argument(
        '--download',
        help='Physically download data. Note this option is not necessary to proceed with statistical analyses, as data can be handled virtually in the program.',
        action='store_true', dest='download', default=False)

    add_cpus(misc)
    add_verbose(misc)

    args =  p.parse_args()

    dlGNSS(args)
    return


## ------------------------------------------------------------ prepFromGUNW.py
def calcDelaysGUNW(iargs: list[str] = None) -> xr.Dataset:

    p = argparse.ArgumentParser(
        description='Calculate a cube of interferometic delays for GUNW files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument(
        '--bucket',
        help='S3 bucket containing ARIA GUNW NetCDF file. Will be ignored if the --file argument is provided.'
    )

    p.add_argument(
        '--bucket-prefix', default='',
        help='S3 bucket prefix containing ARIA GUNW NetCDF file. Will be ignored if the --file argument is provided.'
    )

    p.add_argument(
        '-f', '--file', type=str,
        help='1 ARIA GUNW netcdf file'
    )

    p.add_argument(
        '-m', '--weather-model', default='HRRR', type=str,
        choices=['None'] + ALLOWED_MODELS, help='Weather model.'
    )

    p.add_argument(
        '-uid', '--api_uid', default=None, type=str,
        help='Weather model API UID [uid, email, username], depending on model.'
    )

    p.add_argument(
        '-key', '--api_key', default=None, type=str,
        help='Weather model API KEY [key, password], depending on model.'
    )

    p.add_argument(
        '-interp', '--interpolate-time', default='azimuth_time_grid', type=str,
        choices=['none', 'center_time', 'azimuth_time_grid'],
        help=('How to interpolate across model time steps. Possible options are: '
              '[\'none\', \'center_time\', \'azimuth_time_grid\'] '
              'None: means nearest model time; center_time: linearly across center time; '
              'Azimuth_time_grid: means every pixel is weighted with respect to azimuth time of S1;'
              )
    )

    p.add_argument(
        '-o', '--output-directory', default=os.getcwd(), type=str,
        help='Directory to store results.'
    )

    iargs = p.parse_args(iargs)

    if iargs.interpolate_time not in ['none', 'center_time', 'azimuth_time_grid']:
        raise ValueError('interpolate_time arg must be in [\'none\', \'center_time\', \'azimuth_time_grid\']')

    if iargs.weather_model == 'None':
        # NOTE: HyP3's current step function implementation does not have a good way of conditionally
        #       running processing steps. This allows HyP3 to always run this step but exit immediately
        #       and do nothing if tropospheric correction via RAiDER is not selected. This should not cause
        #       any appreciable cost increase to GUNW product generation.
        print('Nothing to do!')
        return

    if iargs.file and (iargs.weather_model == 'HRRR') and (iargs.interpolate_time == 'azimuth_time_grid'):
        file_name = iargs.file.split('/')[-1]
        gunw_id = file_name.replace('.nc', '')
        if not RAiDER.aria.prepFromGUNW.check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation(gunw_id):
            raise ValueError('The required HRRR data for time-grid interpolation is not available')

    if not iargs.file and iargs.bucket:
        # only use GUNW ID for checking if HRRR available
        if iargs.weather_model == 'HRRR' and (iargs.interpolate_time == 'azimuth_time_grid'):
            gunw_nc_name = iargs.bucket_prefix.split('/')[-1]
            gunw_id = gunw_nc_name.replace('.nc', '')
            if not RAiDER.aria.prepFromGUNW.check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation(gunw_id):
                print('The required HRRR data for time-grid interpolation is not available; returning None and not modifying GUNW dataset')
                return

        # Download file to obtain metadata
        iargs.file = aws.get_s3_file(iargs.bucket, iargs.bucket_prefix, '.nc')
        if not RAiDER.aria.prepFromGUNW.check_weather_model_availability(iargs.file, iargs.weather_model):
            # NOTE: We want to submit jobs that are outside of acceptable weather model range
            #       and still deliver these products to the DAAC without this layer. Therefore
            #       we include this within this portion of the control flow.
            print('Nothing to do because outside of weather model range')
            return
        json_file_path = aws.get_s3_file(iargs.bucket, iargs.bucket_prefix, '.json')
        json_data = json.load(open(json_file_path))
        json_data['metadata'].setdefault('weather_model', []).append(iargs.weather_model)
        json.dump(json_data, open(json_file_path, 'w'))

    elif not iargs.file:
        raise ValueError('Either argument --file or --bucket must be provided')

    # prep the config needed for delay calcs
    path_cfg, wavelength = RAiDER.aria.prepFromGUNW.main(iargs)

    # write delay cube (nc) to disk using config
    # return a list with the path to cube for each date
    cube_filenames = calcDelays([path_cfg])

    assert len(cube_filenames) == 2, 'Incorrect number of delay files written.'

    # calculate the interferometric phase and write it out
    ds = RAiDER.aria.calcGUNW.tropo_gunw_slc(cube_filenames,
                                             iargs.file,
                                             wavelength,
                                             )

    # upload to s3
    if iargs.bucket:
        aws.upload_file_to_s3(iargs.file, iargs.bucket, iargs.bucket_prefix)
        aws.upload_file_to_s3(json_file_path, iargs.bucket, iargs.bucket_prefix)
    return ds


## ------------------------------------------------------------ processDelays.py
def combineZTDFiles():
    '''
    Command-line program to process delay files from RAiDER and GNSS into a single file.
    '''
    from RAiDER.gnss.processDelayFiles import main, combineDelayFiles, create_parser

    p = create_parser()
    args = p.parse_args()

    if not os.path.exists(args.raider_file):
        combineDelayFiles(args.raider_file, loc=args.raider_folder)

    if not os.path.exists(args.gnss_file):
        combineDelayFiles(args.gnss_file, loc=args.gnss_folder, source='GNSS',
                          ref=args.raider_file, col_name=args.column_name)

    if args.gnss_file is not None:
        main(
            args.raider_file,
            args.gnss_file,
            col_name=args.column_name,
            raider_delay=args.raider_column_name,
            outName=args.out_name,
            localTime=args.local_time
        )
