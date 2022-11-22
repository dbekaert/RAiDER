import argparse
import os
import shutil
import sys
import yaml

import RAiDER
from RAiDER.constants import _ZREF, _CUBE_SPACING_IN_M
from RAiDER.logger import logger, logging
from RAiDER.cli.validators import enforce_time, enforce_bbox, parse_dates, get_query_region, get_heights, get_los, enforce_wm

STEP_LIST = [
    'load_weather_model',
    'calculate_delays',
    'download_gnss',
]

STEP_HELP = """Command line options for steps processing with names are chosen from the following list:
{}
In order to use either --start or --dostep, it is necessary that a
previous run was done using one of the steps options to process at least
through the step immediately preceding the starting step of the current run.
""".format(STEP_LIST[0:])


HELP_MESSAGE = """
Command line options for RAiDER processing to calculate tropospheric
delay from a weather model. Default options can be found by running
raiderDelay.py --generate_config.
"""

SHORT_MESSAGE = """
Program to calculate troposphere total delays using a weather model
"""

EXAMPLES = """
Usage examples:
raiderDelay.py -g
raiderDelay.py customTemplatefile.cfg
raiderDelay.py --dostep=load_weather_model
"""

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

DEFAULT_DICT = dict(
        look_dir='right',
        date_start=None,
        date_end=None,
        date_step=None,
        date_list=None,
        time=None,
        end_time=None,
        weather_model=None,
        lat_file=None,
        lon_file=None,
        station_file=None,
        bounding_box=None,
        geocoded_file=None,
        dem=None,
        use_dem_latlon=False,
        height_levels=None,
        height_file_rdr=None,
        ray_trace=False,
        zref=_ZREF,
        cube_spacing_in_m=_CUBE_SPACING_IN_M,  # TODO - Where are these parsed?
        los_file=None,
        los_convention='isce',
        los_cube=None,
        orbit_file=None,
        verbose=True,
        raster_format='GTiff',
        output_directory=os.getcwd(),
        weather_model_directory=os.path.join(
            os.getcwd(),
            'weather_files'
        ),
        output_projection='EPSG:4236',
    )


def create_parser():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = HELP_MESSAGE,
        epilog = EXAMPLES,
    )

    p.add_argument(
        'customTemplateFile', nargs='?',
        help='custom template with option settings.\n' +
        "ignored if the default smallbaselineApp.cfg is input."
    )

    p.add_argument(
        '-g', '--generate_template',
        dest='generate_template',
        action='store_true',
        help='generate default template (if it does not exist) and exit.'
    )

    step = p.add_argument_group('steps processing (start/end/dostep)', STEP_HELP)
    step.add_argument('--start', dest='startStep', metavar='STEP', default=STEP_LIST[0],
                      help='start processing at the named step (default: %(default)s).')
    step.add_argument('--end','--stop', dest='endStep', metavar='STEP',  default=STEP_LIST[-1],
                      help='end processing at the named step (default: %(default)s)')
    step.add_argument('--dostep', dest='doStep', metavar='STEP',
                      help='run processing at the named step only')

    return p


def parseCMD(iargs=None):
    """
    Parse command-line arguments and pass to tropo_delay
    We'll parse arguments and call delay.py.
    """

    p = create_parser()
    args = p.parse_args(args=iargs)

    args.argv = iargs if iargs else sys.argv[1:]

    # default input file
    template_file = os.path.join(
        os.path.dirname(
            RAiDER.__file__
        ),
        'cli', 'raiderDelay.yaml'
    )
    if '-g' in args.argv:
        dst = os.path.join(os.getcwd(), 'raiderDelay.yaml')
        shutil.copyfile(
                template_file,
                dst,
            )

        logger.info('Wrote %s', dst)
        sys.exit(0)

    # check: existence of input template files
    if (not args.customTemplateFile
            and not os.path.isfile(os.path.basename(template_file))
            and not args.generate_template):
        p.print_usage()
        print(EXAMPLES)

        msg = "No template file found! It requires that either:"
        msg += "\n  a custom template file, OR the default template "
        msg += "\n  file 'raiderDelay.yaml' exists in current directory."
        raise SystemExit('ERROR: {}'.format(msg))

    if  args.customTemplateFile:
        # check the existence
        if not os.path.isfile(args.customTemplateFile):
            raise FileNotFoundError(args.customTemplateFile)

        args.customTemplateFile = os.path.abspath(args.customTemplateFile)

    # check which steps to run
    args.runSteps = read_inps2run_steps(args, step_list=STEP_LIST)

    return args


def read_inps2run_steps(inps, step_list):
    """read/get run_steps from input arguments."""
    # check: if start/end/do step input is valid
    for key in ['startStep', 'endStep', 'doStep']:
        value = vars(inps)[key]
        if value and value not in step_list:
            msg = 'Input step not found: {}'.format(value)
            msg += '\nAvailable steps: {}'.format(step_list)
            raise ValueError(msg)

    # currently forcing two delay steps if dostep is NOT specified
        # check: ignore --start/end input if --dostep is specified; re-implement
    if inps.doStep:
        # inps.startStep = inps.doStep
        # inps.endStep = inps.doStep
        run_steps    = [inps.doStep]

    else:
        # get list of steps to run
        idx0 = step_list.index(inps.startStep)
        idx1 = step_list.index(inps.endStep)
        if idx0 > idx1:
            msg = 'start step "{}" CAN NOT be after the end step "{}"'.format(inps.startStep, inps.endStep)
            raise ValueError(msg)

        run_steps = step_list[idx0:idx1]

    # print mssage - processing steps
    print('Run routine processing with {} on steps: {}'.format(os.path.basename(__file__), run_steps))
    # print('Remaining steps: {}'.format(step_list[idx0+1:]))
    print('-'*50)

    return run_steps


def read_template_file(fname):
    """
    Read the template file into a dictionary structure.
    Parameters: fname      - str, full path to the template file
                delimiter  - str, string to separate the key and value
                skip_chars - list of str, skip certain charaters in values
    Returns:    template   - dict, file content
    Examples:   template = read_template('raiderDelay.yaml')

    Modified from MintPy's 'read_template'
    """
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
    template = DEFAULT_DICT
    for key, value in params.items():
        if key == 'runtime_group':
            for k, v in value.items():
                if v is not None:
                    template[k] = v
        if key == 'weather_model':
            template[key]= enforce_wm(value)
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
            template['los'] = get_los(AttributeDict(value))
        if key == 'look_dir':
            if value.lower() not in ['right', 'left']:
                raise ValueError(f"Unknown look direction {value}")
            template['look_dir'] = value.lower()

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
