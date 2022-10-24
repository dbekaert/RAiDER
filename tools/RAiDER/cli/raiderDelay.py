import argparse
import os
import sys

import RAiDER
from RAiDER.checkArgs import checkArgs
from RAiDER.cli.parser import add_bbox, add_out, add_verbose
from RAiDER.cli.validators import DateListAction, date_type, time_type
from RAiDER.constants import _ZREF
from RAiDER.logger import logger, logging
from RAiDER.models.allowed import ALLOWED_MODELS

STEP_LIST = [
    'load_weather_model',
    'calculate_delays',
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
    utm_zone=None,
    grid_x=None,
    grid_y=None,
    dem=None,
    use_dem_latlon=False,
    height_levels=None,
    ray_trace=False,
    zref=_ZREF,
    los_file=None,
    los_convention='isce',
    los_cube=None,
    orbit_file=None,
    verbose=True,
    raster_format='GTiff'
    output_directory=os.getcwd(),
    weather_model_directory=os.path.join(output_directory,'weather_files'),
    output_projection='EPSG:4236',
)

def create_parser():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(
        help = SHORT_MESSAGE,
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
        '-g', 
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
        'tools', os.sep, 
        'RAiDER', os.sep, 
        'cli', os.sep, 
        'raiderDelay.cfg'
    )

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

    # check: ignore --start/end input if --dostep is specified
    if inps.doStep:
        inps.startStep = inps.doStep
        inps.endStep = inps.doStep

    # get list of steps to run
    idx0 = step_list.index(inps.startStep)
    idx1 = step_list.index(inps.endStep)
    if idx0 > idx1:
        msg = 'start step "{}" CAN NOT be after the end step "{}"'.format(inps.startStep, inps.endStep)
        raise ValueError(msg)
    run_steps = step_list[idx0:idx1+1]

    # empty the step list if:
    # a) -g OR
    if inps.generate_template:
        run_steps = []

    # print mssage - processing steps
    print('Run routine processing with {} on steps: {}'.format(os.path.basename(__file__), run_steps))
    print('Remaining steps: {}'.format(step_list[idx0+1:]))
    print('-'*50)

    return run_steps


def read_template_file(fname, delimiter='=', skip_chars=None):
    """
    Read the template file into a dictionary structure.
    Parameters: fname      - str, full path to the template file
                delimiter  - str, string to separate the key and value
                skip_chars - list of str, skip certain charaters in values
    Returns:    template   - dict, file content
    Examples:   template = read_template('raiderDelay.yaml')

    Modified from MintPy's 'read_template'
    """
    if skip_chars and isinstance(skip_chars, str):
        skip_chars = [skip_chars]

    # read input text file / string
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    elif isinstance(fname, str):
        lines = fname.split('\n')
    else:
        raise ValueError('{} is not a valid template file name'.format(fname))

    lines = [x.strip() for x in lines]

    # parse line by line
    template = DEFAULT_DICT
    for line in lines:
        # split on the 1st occurrence of delimiter
        c = [i.strip() for i in line.split(delimiter, 1)]

        # ignore commented lines or those without variables
        if len(c) >= 2 and not line.startswith(('%', '#', '!')):
            key = c[0]
            value = str.replace(c[1], '\n', '').split("#")[0].strip()
            value = os.path.expanduser(value)  # translate ~ symbol
            value = os.path.expandvars(value)  # translate env variables

            # skip certain characters by replacing them with empty str
            if skip_chars:
                for skip_char in skip_chars:
                    value.replace(skip_char, '')

            if key == 'model':
                value = value.upper().replace("-", "")
        
        template[key]=value

    return template


##########################################################################
def main(iargs=None):
    # parse
    inps = parseCMD(iargs)
    
    # Read the template file
    args = read_template_file(inps.customTemplateFile)
    
    # Argument checking
    args = checkArgs(args)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Run the list of commands
    from RAiDER.runProgram import _tropo_delay

    # run
    _tropo_delay(args)


###########################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])