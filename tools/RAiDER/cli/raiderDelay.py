import argparse
import copy
import multiprocessing
import os

import numpy as np

from textwrap import dedent

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
raiderDelay.py 
raiderDelay.py 
"""

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


    # datetime = p.add_argument_group('Datetime')
    # datetime.add_argument(
    #     '--date', dest='dateList',
    #     help=dedent("""\
    #         Date to calculate delay.
    #         Can be a single date, a list of two dates (earlier, later) with 1-day interval, or a list of two dates and interval in days (earlier, later, interval).
    #         Example accepted formats:
    #            YYYYMMDD or
    #            YYYYMMDD YYYYMMDD
    #            YYYYMMDD YYYYMMDD N
    #         """),
    #     nargs="+",
    #     action=DateListAction,
    #     type=date_type,
    #     required=True
    # )

    # datetime.add_argument(
    #     '--time', dest='time',
    #     help=dedent('''\
    #     Calculate delay at this time.
    #     Example formats:
    #        THHMMSS,
    #        HHMMSS, or
    #        HH:MM:SS'''),
    #     type=time_type, required=True)

    # # Area
    # area = p.add_argument_group('Area of Interest (Supply one)').add_mutually_exclusive_group(required=True)
    # area.add_argument(
    #     '--latlon',
    #     '-ll',
    #     nargs=2,
    #     dest='query_area',
    #     help='GDAL-readable latitude and longitude raster files (2 single-band files)',
    #     metavar=('LAT', 'LONG')
    # )
    # add_bbox(area)
    # area.add_argument(
    #     '--station_file',
    #     default=None,
    #     type=str,
    #     dest='query_area',
    #     help=('CSV file with a list of stations, containing at least '
    #           'the columns "Lat" and "Lon"')
    # )

    # # Line of sight
    # los = p.add_argument_group(
    #     'Specify a Line-of-sight or state vector file. If neither argument is supplied, the Zenith delay will be returned'
    # ).add_mutually_exclusive_group()
    # los.add_argument(
    #     '--lineofsight', '-l',
    #     help='GDAL-readable two-band line-of-sight file (B1: inclination, B2: heading)',
    #          metavar='LOS', default=None)
    # los.add_argument(
    #     '--statevectors', '-s', default=None, metavar='SV',
    #     help='An ESA orbit file or text file containing state vectors specifying '
    #          'the orbit of the sensor.')

    # # heights
    # heights = p.add_argument_group('Height data. Default is ground surface for specified lat/lons, height levels otherwise')
    # heights.add_argument(
    #     '--dem', '-d', default=None,
    #     help="""Specify a DEM to use with lat/lon inputs.""")
    # heights.add_argument(
    #     '--heightlvs',
    #     help=("""A space-deliminited list of heights"""),
    #     default=None, nargs='+', type=float)

    # # Weather model
    # p.add_argument(
    #     '--model',
    #     help="Weather model to access.",
    #     type=lambda s: s.upper().replace("-", ""),
    #     choices=ALLOWED_MODELS,
    #     default='ERA5T')

    add_verbose(p)

    return p


def parseCMD():
    """
    Parse command-line arguments and pass to tropo_delay
    We'll parse arguments and call delay.py.
    """

    p = create_parser()
    args = p.parse_args()

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
        parser.print_usage()
        print(EXAMPLE)
        
        msg = "No template file found! It requires that a:"
        msg += "\n  a custom template file, OR"
        msg += "\n  2) the default template file 'raiderDelay.cfg' "
        msg += "\n exists in current directory."
        raise SystemExit('ERROR: {}'.format(msg))

    if  args.customTemplateFile:
        # check the existence
        if not os.path.isfile(args.customTemplateFile):
            raise FileNotFoundError(args.customTemplateFile)
        
        args.customTemplateFile = os.path.abspath(args.customTemplateFile)

    # check which steps to run
    args.runSteps = read_inps2run_steps(args, step_list=STEP_LIST)

    # Argument checking
    args = checkArgs(args, p)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

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
    Examples:   template = read_template('KyushuAlosAT424.txt')
                template = read_template('smallbaselineApp.cfg')

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
    template = {}
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

            if value != '':
                template[key] = value

    return template


##########################################################################
def main(iargs=None):
    # parse
    inps = parseCMD(iargs)

    # import
    from RAiDER.runProgram import _tropo_delay

    # run
    _tropo_delay(inps)


###########################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])