import os
import argparse
from importlib.metadata import entry_points
from textwrap import dedent
from RAiDER.cli.parser import add_cpus, add_out, add_verbose
from RAiDER.cli.validators import DateListAction, date_type


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


def calcDelays(iargs=None):
    """Parse command line arguments using argparse."""
    import RAiDER
    from RAiDER.runDelays import main as runDelay
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

    p.add_argument(
        '--download-only',
        action='store_true',
        help='only download a weather model.'
    )

    args = p.parse_args(args=iargs)

    args.argv = iargs if iargs else os.sys.argv[1:]

    # default input file
    template_file = os.path.join(
        os.path.dirname(
            RAiDER.__file__
        ),
        'cli', 'raider.yaml'
    )
    if '-g' in args.argv:
        dst = os.path.join(os.getcwd(), 'raider.yaml')
        shutil.copyfile(
                template_file,
                dst,
            )

        logger.info('Wrote %s', dst)
        os.sys.exit(0)

    # check: existence of input template files
    if (not args.customTemplateFile
            and not os.path.isfile(os.path.basename(template_file))
            and not args.generate_template):
        p.print_usage()
        print(EXAMPLES)

        msg = "No template file found! It requires that either:"
        msg += "\n  a custom template file, OR the default template "
        msg += "\n  file 'raider.yaml' exists in current directory."
        raise SystemExit(f'ERROR: {msg}')

    if  args.customTemplateFile:
        # check the existence
        if not os.path.isfile(args.customTemplateFile):
            raise FileNotFoundError(args.customTemplateFile)

        args.customTemplateFile = os.path.abspath(args.customTemplateFile)

    runDelay(args)

    return p


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


def main():
    parser = argparse.ArgumentParser(prefix_chars='+', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '++process', choices=['runDelays', 'raiderDownloadGNSS'], default='runDelays',
        help='Select the HyP3 entrypoint to use'
    )
    args, unknowns = parser.parse_known_args()

    os.sys.argv = [args.process, *unknowns]
    # FIXME: this gets better in python 3.10
    # (process_entry_point,) = entry_points(group='console_scripts', name=args.process)

    process_entry_point = [ep for ep in entry_points()['console_scripts'] if ep.name.startswith(args.process)][0]
    os.sys.exit(
        process_entry_point.load()()
    )


if __name__ == '__main__':
    main()
