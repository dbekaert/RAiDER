import argparse
from textwrap import dedent

from RAiDER.checkArgs import checkArgs
from RAiDER.cli.parser import add_bbox, add_out, add_verbose
from RAiDER.cli.validators import DateListAction, date_type, time_type
from RAiDER.constants import _ZREF
from RAiDER.delay import tropo_delay, weather_model_debug
from RAiDER.logger import *
from RAiDER.models.allowed import ALLOWED_MODELS
import numpy as np
import copy



def create_parser():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent("""\
            Calculate tropospheric delay from a weather model.
            Usage examples:
            raiderDelay.py --date 20200103 --time 23:00:00 -b 39 40 -79 -78 --model ERA5 --zref 15000 -v
            raiderDelay.py --date 20200103 --time 23:00:00 -b 39 40 -79 -78 --model ERA5 --zref 15000 --heightlvs 0 100 200 -v
            raiderDelay.py --date 20200103 --time 23:00:00 --latlon test/scenario_1/geom/lat.dat test/scenario_1/geom/lon.dat --model ERA5 --zref 20000 -v --out test/scenario_1/
            """)
    )

    datetime = p.add_argument_group('Datetime')
    datetime.add_argument(
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

    datetime.add_argument(
        '--time', dest='time',
        help=dedent('''\
        Calculate delay at this time.
        Example formats:
           THHMMSS,
           HHMMSS, or
           HH:MM:SS'''),
        type=time_type, required=True)

    # Area
    area = p.add_argument_group('Area of Interest (Supply one)').add_mutually_exclusive_group(required=True)
    area.add_argument(
        '--latlon',
        '-ll',
        nargs=2,
        dest='query_area',
        help='GDAL-readable latitude and longitude raster files (2 single-band files)',
        metavar=('LAT', 'LONG')
    )
    add_bbox(area)
    area.add_argument(
        '--station_file',
        default=None,
        type=str,
        dest='query_area',
        help=('CSV file with a list of stations, containing at least '
              'the columns "Lat" and "Lon"')
    )

    # Line of sight
    los = p.add_argument_group(
        'Specify a Line-of-sight or state vector file. If neither argument is supplied, the Zenith delay will be returned'
    ).add_mutually_exclusive_group()
    los.add_argument(
        '--lineofsight', '-l',
        help='GDAL-readable two-band line-of-sight file (B1: inclination, B2: heading)',
             metavar='LOS', default=None)
    los.add_argument(
        '--statevectors', '-s', default=None, metavar='SV',
        help='An ESA orbit file or text file containing state vectors specifying '
             'the orbit of the sensor.')

    # heights
    heights = p.add_argument_group('Height data. Default is ground surface for specified lat/lons, height levels otherwise')
    heights.add_argument(
        '--dem', '-d', default=None,
        help="""Specify a DEM to use with lat/lon inputs.""")
    heights.add_argument(
        '--heightlvs',
        help=("""A space-deliminited list of heights"""),
        default=None, nargs='+', type=float)

    # Weather model
    weather = p.add_argument_group("Weather model. See documentation for details")
    weather.add_argument(
        '--model',
        help="Weather model option to use.",
        type=lambda s: s.upper().replace("-", ""),
        choices=ALLOWED_MODELS,
        default='ERA5T')
    weather.add_argument(
        '--files',
        help="""OUT/PLEV or HDF5 file(s) """,
        default=None, nargs='+', type=str, metavar="FILES")
    weather.add_argument(
        '--weatherFiles', '-w',
        help='Directory location of/to write weather model files',
        default=None, dest='wmLoc')

    misc = p.add_argument_group("Run parameters")
    misc.add_argument(
        '--zref', '-z',
        help='Height limit when integrating (meters) (default: {} m)'.format(_ZREF),
        type=float,
        default=_ZREF)
    misc.add_argument(
        '--outformat',
        help='GDAL-compatible file format if surface delays are requested.',
        default=None)

    add_out(misc)

    misc.add_argument(
        '--download_only',
        help='Download weather model only without processing? Default False',
        action='store_true', dest='download_only', default=False)

    add_verbose(misc)

    return p


def parseCMD():
    """
    Parse command-line arguments and pass to tropo_delay
    We'll parse arguments and call delay.py.
    """

    p = create_parser()
    args = p.parse_args()

    # Argument checking
    args = list(checkArgs(args, p))

    download_only, verbose    = args[-4:-2]
    idxT, idxW, idxH            = 10, 14, 15
    times, wetNames, hydroNames = args.pop(idxT), args.pop(idxW-1), args.pop(idxH-2)

    if verbose: logger.setLevel(logging.DEBUG)

    if download_only:
        import multiprocessing
        max_threads = multiprocessing.cpu_count()
        nt     = 5
        if nt == 'all':
            nt = max_threads
        else:
            nt = int(nt)
        nt     = nt if nt < max_threads else max_threads


        chunked_args  = [] # dictionary for each process
        allTimesFiles = zip(times, wetNames, hydroNames)
        allTimesFiles_chunk = np.array_split(list(allTimesFiles), nt)
        lst_new_args  = []

        for chunk in allTimesFiles_chunk:
            if chunk.size == 0: continue
            times, wetNames, hydroNames = chunk.transpose()
            args_copy = copy.deepcopy(args)
            args_copy.insert(idxT, times.tolist())
            args_copy.insert(idxW, wetNames.tolist())
            args_copy.insert(idxH, hydroNames.tolist())
            lst_new_args.append(args_copy)

        _tropo_delay(lst_new_args[0])
        # import pdb; pdb.set_trace()
        with multiprocessing.Pool(len(lst_new_args)) as pool:
            pool.map(_tropo_delay, lst_new_args)
        return

    else:
        p = create_parser()
        args = p.parse_args()
        los, lats, lons, ll_bounds, heights, flag, weather_model, wmLoc, zref, outformat, \
        times, out, download_only, verbose, \
        wetNames, hydroNames = checkArgs(args, p)

        # Loop over each datetime and compute the delay
        for t, wfn, hfn in zip(times, wetNames, hydroNames):
            try:
                import pdb; pdb.set_trace()
                (_, _) = tropo_delay(los, lats, lons, ll_bounds, heights, flag, weather_model, wmLoc, zref,
                                     outformat, t, out, download_only, wfn, hfn)

            except RuntimeError:
                logger.exception("Date %s failed", t)
                continue

def _tropo_delay(chunk_params):
    chunk_params = copy.deepcopy(chunk_params)
    chunk_params.pop(-3) # no verbose parm
    times = chunk_params[-5]
    wetNames = chunk_params[-2]
    hydroNames = chunk_params[-1]
    if len(times) < 2:
        try:
            chunk_params[-1]=hydroNames[0]
            chunk_params[-2]=wetNames[0]
            chunk_params[-5]=times[0]
            (_, _) = tropo_delay(*chunk_params)
        except RuntimeError:
            logger.exception("Date %s failed", t)

    else:
        for t, wfn, hfn in zip(times, wetNames, hydroNames):
            try:
                chunk_params[-1]=hfn
                chunk_params[-2]=wfn
                chunk_params[-5]=t
                (_, _) = tropo_delay(*chunk_params)
            except RuntimeError:
                logger.exception("Date %s failed", t)
                continue
    return

def parseCMD_weather_model_debug():
    """
    Parse command-line arguments and pass to prepareWeatherModel
    We'll parse arguments and call delay.py.
    """

    p = create_parser()
    args = p.parse_args()

    # Argument checking
    los, lats, lons, ll_bounds, heights, flag, weather_model, wmLoc, zref, outformat, \
        times, out, download_only, verbose, \
        wetNames, hydroNames = checkArgs(args, p)

    if verbose:
        logger.setLevel(logging.DEBUG)

    # Loop over each datetime
    for t in times:
        try:
            weather_model_debug(los, lats, lons, ll_bounds, weather_model, wmLoc, zref, t, out, download_only)

        except RuntimeError:
            logger.exception("Date %s failed", t)
            continue
