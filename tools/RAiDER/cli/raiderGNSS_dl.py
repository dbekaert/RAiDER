import argparse
import os
import shutil
import sys
import yaml

import RAiDER
from RAiDER.constants import _ZREF, _CUBE_SPACING_IN_M
from RAiDER.logger import logger, logging
from RAiDER.cli.validators import enforce_time, enforce_bbox, parse_dates, get_query_region, get_heights, get_los, enforce_wm


HELP_MESSAGE = """
  Command line options for RAiDER to download GNSS stations from UNR (MIDAS).
"""

EXAMPLES = """
Usage examples:
raiderDelay.py -g
raiderDelay.py customTemplatefile.cfg
raiderDelay.py --dostep=load_weather_model
"""


def create_parser_GNSS():
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


    return p


def parseCMD_GNSS(iargs=None):
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
        shutil.copyfile(
                template_file,
                os.path.join(os.getcwd(), 'raiderDelay.yaml'),
            )
        sys.exit(0)

    # check: existence of input template files
    if (not args.customTemplateFile
            and not os.path.isfile(os.path.basename(template_file))
            and not args.generate_template):
        p.print_usage()
        print(EXAMPLES)

        msg = "No template file found! It requires that either:"
        msg += "\n  a custom template file, OR the default template "
        msg += "\n  file 'raiderGNSS_dl.yaml' exists in current directory."
        raise SystemExit('ERROR: {}'.format(msg))

    if  args.customTemplateFile:
        # check the existence
        if not os.path.isfile(args.customTemplateFile):
            raise FileNotFoundError(args.customTemplateFile)

        args.customTemplateFile = os.path.abspath(args.customTemplateFile)

    return args


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
            template['aoi'] = get_query_region(AttributeDict(value))
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
