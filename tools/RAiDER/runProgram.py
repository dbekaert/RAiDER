import argparse

from RAiDER.constants import _ZREF
from RAiDER.utilFcns import parse_date, parse_time


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
    p = argparse.ArgumentParser(
          formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Calculate tropospheric delay from a weather model.
Usage examples:
raiderDelay.py --date 20200103 --time 23:00:00 -b 40 -79 39 -78 --model ERA5 --zref 15000 -v
raiderDelay.py --date 20200103 --time 23:00:00 -b 40 -79 39 -78 --model ERA5 --zref 15000 --heightlvs 0 100 200 -v
raiderDelay.py --date 20200103 --time 23:00:00 --latlon test/scenario_1/geom/ERA5_Lat_2018_01_01_T00_00_00.dat test/scenario_1/geom/ERA5_Lon_2018_01_01_T00_00_00.dat --model ERA5 --zref 20000 -v --out test/scenario_1/
""")

    datetime = p.add_argument_group('Datetime')
    datetime.add_argument(
        '--date',dest='dateList',
        help="""Date to calculate delay.
Can be a single date or a comma-separated list of two dates (earlier, later).
Example accepted formats:
   YYYYMMDD or
   YYYYMMDD,YYYYMMDD
""",
        type=read_date, required=True)
    datetime.add_argument(
        '--time', dest = 'time',
        help='''Calculate delay at this time.
Example formats:
   THHMMSS,
   HHMMSS, or
   HH:MM:SS''',
        type=parse_time, required=True)

    # Area
    area = p.add_argument_group('Area of Interest (Supply one)')
    area.add_argument(
        '--latlon', '-ll', nargs=2,default = None,
        help=('GDAL-readable latitude and longitude raster files (2 single-band files)'),
        metavar=('LAT', 'LONG'))
    area.add_argument(
        '--BBOX', '-b', nargs=4, dest='bounding_box',
        help="""Bounding box""",
        metavar=('N', 'W', 'S', 'E'))
    area.add_argument(
        '--station_file', default = None, type=str, dest='station_file',
        help=('CSV file with a list of stations, containing at least '
              'the columns "Lat" and "Lon"'))

    # Line of sight
    los = p.add_argument_group('Line-of-sight options. If neither argument is supplied, the Zenith delay will be returned')
    los.add_argument(
        '--lineofsight', '-l',
        help='GDAL-readable two-band line-of-sight file (B1: inclination, B2: heading)',
             metavar='LOS', default=None)
    los.add_argument(
        '--statevectors', '-s', default=None, metavar='SV',
        help=('An ESA orbit file or text file containing state vectors specifying ' \
              'the orbit of the sensor.'))

    # heights
    heights = p.add_argument_group('Height data. Default is ground surface for specified lat/lons, height levels otherwise')
    heights.add_argument(
        '--dem', '-d', default=None,
        help="""Specify a DEM to use with lat/lon inputs.""")
    heights.add_argument(
        '--heightlvs',
        help=("""A space-deliminited list of heights"""),
        default=None,nargs='+', type=float)

    # Weather model
    weather = p.add_argument_group("Weather model. See documentation for details")
    weather.add_argument(
        '--model',
        help="""Weather model option to use: ERA5/HRRR/MERRA2/NARR/WRF/HDF5. """,
        default='ERA-5T')
    weather.add_argument(
        '--files',
        help="""OUT/PLEV or HDF5 file(s) """,
        default=None, nargs='+', type=str, metavar="FILES")

    weather.add_argument(
        '--weatherModelFileLocation', '-w',
        help='Directory location of/to write weather model files',
        default=None, dest='wmLoc')


    misc = p.add_argument_group("Run parameters")
    misc.add_argument(
        '--zref', '-z',
        help=('Height limit when integrating (meters) (default: {}s)'.format(_ZREF)),
        default=_ZREF)
    misc.add_argument(
        '--outformat',
        help='GDAL-compatible file format if surface delays are requested.',
        default=None)

    misc.add_argument(
        '--out',
        help='Output file directory',
        default='.')

    misc.add_argument(
        '--download_only',
        help='Download weather model only without processing? Default False',
        action='store_true',dest='download_only', default = False)

    misc.add_argument(
        '--verbose', '-v',
        help='Run in verbose (debug) mode? Default False',
        action='store_true',dest='verbose', default = False)

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
    los, lats, lons, ll_bounds, heights, flag, weather_model, wmLoc, zref, outformat, \
         times, out, download_only, verbose, \
         wetNames, hydroNames= checkArgs(args, p)

    # Loop over each datetime and compute the delay
    for t, wfn, hfn in zip(times, wetNames, hydroNames):
        try:
            (_,_) = tropo_delay(los, lats, lons, ll_bounds, heights, flag, weather_model, wmLoc, zref,
               outformat, t, out, download_only, verbose, wfn, hfn)

        except RuntimeError as e:
            print('Date {} failed'.format(t))
            print(e)
            continue
