import os
import numpy as np

import RAiDER.utilFcns
from RAiDER.utilFcns import parse_date, parse_time
from RAiDER.constants import _ZREF

def read_date(s):
    '''
    Read and parse an input date or datestring
    '''
    import datetime
    try:
        date1, date2 = [parse_date(d) for d in s.split(',')]
        dateList = [date1 + k*datetime.timdelta(days=1) for k in range((date2 - date1).days+1)]
        return dateList
    except ValueError:
        date = parse_date(s)
        return [date]


def parse_args():
    """Parse command line arguments using argparse."""
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Calculate tropospheric delay from a weather model. 
Usage examples: 
raiderDelay.py --date 20200103 --time 23:00:00 -b 40 -79 39 -78 --model ERA5 --zref 15000 -v
raiderDelay.py --date 20200103 --time 23:00:00 -b 40 -79 39 -78 --model ERA5 --zref 15000 --heightlvs 0 100 200 -v 
""")

    p.add_argument(
        '--date',dest='dateList',
        help="""Fetch weather model data for a given date or date range.
Can be a single date or a comma-separated list of two dates (earlier, later) in the ISO 8601 format
Example accepted formats: 
   YYYYMMDD or
   YYYYMMDD,YYYYMMDD
""",
        type=read_date, required=True)
    p.add_argument(
        '--time', dest = 'time',
        help='''Fetch weather model data at this time of day. 
Example accepted formats: 
   THHMMSS,
   HHMMSS, or
   HH:MM:SS''',
        type=parse_time, required=True)

    # Area
    area = p.add_mutually_exclusive_group(required=True)
    area.add_argument(
        '--area', '-a', nargs=2,default = None,
        help=('GDAL-readable longitude and latitude files to specify the '
              'region over which to calculate delay. Delay will be '
              'calculated at weather model nodes if unspecified'),
        metavar=('LAT', 'LONG'))
    area.add_argument(
        '--BBOX', '-b', nargs=4, dest='bounding_box',
        help="""Bounding box over which to downloaded the weather model, given as N W S E. 
If a bounding box is supplied, the delays will be calculated only at the x/y grid nodes of the 
weather model that is selected. 
""",
        metavar=('N', 'W', 'S', 'E'))
    area.add_argument(
        '--station_file',default = None, type=str, dest='station_file',
        help=('CSV file containing a list of stations, with at least '
              'the columns "Lat" and "Lon"'))

    # Line of sight
    los = p.add_mutually_exclusive_group()
    los.add_argument(
        '--lineofsight', '-l',
        help='GDAL-readable line-of-sight file. If a LOS or statevector' \
             ' file is not specified, will return the Zenith delay',
             metavar='LOS', default=None)
    los.add_argument(
        '--statevectors', '-s', default=None, metavar='SV',
        help=('An ESA orbit file or text file containing state vectors specifying ' \
              'the orbit of the sensor. If a LOS or statevector file is not specified,' \
              ' will return the Zenith delay'))

    # heights
    heights = p.add_mutually_exclusive_group()
    heights.add_argument(
        '--dem', '-d', default=None,
        help="""DEM file. If not specified but lat/lon data is provided, an SRTM DEM will be downloaded.
If no lat/lon data is provided, the weather model grid nodes will be used.
""")
#    heights.add_argument(
#        '--weather_mnodel_nodes', default=False,
#        help='If True, will use the heights of the weather model nodes')
    heights.add_argument(
        '--heightlvs', default=None,
        help=("""If lat/lon data is provided, the weather delay will be 
calculated at each of these heights (must be specified in meters) for the specified area.
If no lat/lon data is provided, the lat/lon locations of the weather 
model grid nodes will be used with the specified height levels.
"""),
        nargs='+', type=float)

    # Weather model
    p.add_argument(
        '--model',
        help="""Weather model product to use. 
Options:
    ERA5 (Default option, global, 30 km spacing)
    HRRR (Continental US only, 3 km spacing)
    MERRA2 ()
    NARR ()
    WRF (requires providing "OUT" and "PLEV" files with the --files argument)
    HDF5 (provided as an HDF5 file, see documentation for details)
""",
        default='ERA-5')
    p.add_argument(
        '--files', nargs='+', type=str,
        help="""
If using WRF model files, list the OUT file and PLEV file separated by a space
If using an HDF5 file, simply provide the filename
""", default=None, metavar="FILES")

    p.add_argument(
        '--weatherModelFileLocation', '-w', dest='wmLoc',
        help='Directory location of/to write weather model files',
        default='weather_files')

    # Height max
    p.add_argument(
        '--zref', '-z',
        help=('Height limit when integrating (meters) '
              '(default: %(default)s)'),
        type=int, default=_ZREF)

    p.add_argument(
        '--outformat', help='Specify GDAL-compatible format if surface delays are requested.',
        default=None)

    p.add_argument('--out', help='Output file directory', default='.')

    p.add_argument('--download_only', action='store_true',dest='download_only', default = False, help='Download weather model only without processing? Default False')

    p.add_argument('--verbose', '-v', action='store_true',dest='verbose', default = False, help='Run in verbose (debug) mode? Default False')

    p.print_usage()

    return p.parse_args(), p


def parseCMD():
    """
    Parse command-line arguments and pass to tropo_delay
    We'll parse arguments and call delay.py.
    """
    from RAiDER.checkArgs import checkArgs
    from RAiDER.delay import tropo_delay

    args, p = parse_args()

    # Argument checking
    los, lats, lons, heights, flag, weather_model, wmLoc, zref, outformat, \
         times, out, download_only, verbose, \
         wetNames, hydroNames= checkArgs(args, p)

    # Loop over each datetime and compute the delay
    for t, wfn, hfn in zip(times, wetNames, hydroNames):
        try:
            (_,_) = tropo_delay(los, lats, lons, heights, flag, weather_model, wmLoc, zref,
               outformat, t, out, download_only, verbose, wfn, hfn)
        except RuntimeError as e:
            print('Date {} failed'.format(t))
            print(e)
            continue

