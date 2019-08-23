import os
import numpy as np

import RAiDER.delay as delay
from RAiDER.util import makeDelayFileNames as mdf, mkdir
from RAiDER.checkArgs import checkArgs
import RAiDER.llreader as llr


def read_date(s):
    """
    Read a date from a string in pseudo-ISO 8601 format.
    """
    import datetime
    import itertools
    year_formats = (
        '%Y-%m-%d',
        '%Y%m%d',
        '%Y-%m',
        '%Y',  # I don't think anyone would ever want just a year
    )
    time_formats = (
        '',
        'T%H:%M:%S.%f',
        'T%H:%M:%S',
        'T%H%M%S.%f',
        'T%H%M%S',
        'T%H:%M',
        'T%H%M',
        'T%H',
    )
    timezone_formats = (
        '',
        'Z',
        '%z',
    )
    all_formats = map(
        ''.join,
        itertools.product(year_formats, time_formats, timezone_formats))
    date = None
    for date_format in all_formats:
        try:
            date = datetime.datetime.strptime(s, date_format)
        except ValueError:
            continue
    if date is None:
        raise ValueError(
            'Unable to coerce {} to a date. Try %Y-%m-%dT%H:%M:%S.%f%z'.format(s))

    return date


def parse_args():
    """Parse command line arguments using argparse."""
    import argparse
    p = argparse.ArgumentParser(
        description='Calculate tropospheric delay from a weather model')

    p.add_argument(
        '--time',
        help='Fetch weather model data at this (ISO 8601 format) time',
        type=read_date, required=True)

    # Line of sight
    los = p.add_mutually_exclusive_group()
    los.add_argument(
        '--lineofsight', '-l',
        help='GDAL-readable line-of-sight file',
        metavar='LOS', default=None)
    los.add_argument(
        '--statevectors', '-s', default=None,
        help=('An ISCE XML or shelve file containing state vectors specifying '
              'the orbit of the sensor'))

    # Area
    area = p.add_mutually_exclusive_group()
    area.add_argument(
        '--area', '-a', nargs=2,default = None,
        help=('GDAL-readable longitude and latitude files to specify the '
              'region over which to calculate delay. Delay will be '
              'calculated at weather model nodes if unspecified'),
        metavar=('LAT', 'LONG'))

    # model BBOX
    area.add_argument(
        '--modelBBOX', '-modelbb', nargs=4, dest='bounding_box',
        help='BBOX of the model to be downloaded, given as N W S E, if not given defaults in following order: lon-lat derived BBOX, or full world',
        metavar=('N', 'W', 'S', 'E'))
    area.add_argument(
        '--station_file',default = None, type=str, dest='station_file',
        help=('CSV file containing a list of stations, with at least '
              'the columns "Lat" and "Lon"'))

    # heights
    heights = p.add_mutually_exclusive_group()
    heights.add_argument(
        '--dem', '-d', default=None,
        help='DEM file. DEM will be downloaded if not specified')
    heights.add_argument(
        '--heightlvs', default=None,
        help=('Delay will be calculated at each of these heights across '
              'all of the specified area'),
        nargs='+', type=float)

    # Weather model
    wm = p.add_mutually_exclusive_group()
    wm.add_argument(
        '--model',
        help='Weather model to use',
        default='ERA-5')
    wm.add_argument(
        '--pickleFile',
        help='Pickle file to load',
        default=None)
    wm.add_argument(
        '--wmnetcdf',
        help=('Weather model netcdf file. Should have q, t, z, lnsp as '
              'variables'))

    p.add_argument(
        '--weatherModelFileLocation', '-w', dest='wmLoc',
        help='Directory location of/to write weather model files',
        default='weather_files')

    wrf = p.add_argument_group(
        title='WRF',
        description='Arguments for when --model WRF is specified')
    wrf.add_argument(
        '--wrfmodelfiles', nargs=2,
        help='WRF model files',
        metavar=('OUT', 'PLEV'))

    # Height max
    p.add_argument(
        '--zref', '-z',
        help=('Height limit when integrating (meters) '
              '(default: %(default)s)'),
        type=int, default=15000)

    p.add_argument(
        '--outformat', help='Output file format; GDAL-compatible for DEM, HDF5 for height levels',
        default=None)

    p.add_argument('--out', help='Output file directory', default='.')

    p.add_argument('--no_parallel', '-p', action='store_true',dest='no_parallel', default = False, help='Do not run operation in parallel? Default False. Recommend only True for verbose (debug) mode')

    p.add_argument('--download_only', action='store_true',dest='download_only', default = False, help='Download weather model only without processing? Default False')

    p.add_argument('--verbose', '-v', action='store_true',dest='verbose', default = False, help='Run in verbose (debug) mode? Default False')

    return p.parse_args(), p


def writeDelays(flag, wetDelay, hydroDelay, time, los, 
                out, outformat, weather_model_name, 
                proj = None, gt = None):
    '''
    Write the delay numpy arrays to files in the format specified 
    '''
    ndv = 0. # no-data value

    # Use zero for nodata
    wetDelay[np.isnan(wetDelay)] = ndv
    hydroDelay[np.isnan(hydroDelay)] = ndv

    # Do different things, depending on the type of input
    
    wetFilename, hydroFilename = \
          mdf(time, los, outformat, weather_model_name, out)

    if flag=='station_file':
       from RAiDER.util import writeResultsToNETCDF as w2nc
       w2nc(wetDelay, wetFilename, noDataValue = ndv, 
                       fmt=outformat, proj=proj, gt=gt)
    else:
       from RAiDER.util import writeArrayToRaster as watr
       watr(wetDelay, wetFilename, noDataValue = ndv, 
                       fmt = outformat, proj = proj, gt = gt)
       watr(hydroDelay, hydroFilename, noDataValue = ndv, 
                       fmt = outformat, proj = proj, gt = gt)


def main():
    """tropo_delay main function.

    We'll parse arguments and call delay.py.
    """
    args, p = parse_args()

    mkdir(os.path.join(args.out, 'geom'))
    mkdir(os.path.join(args.out, 'weather_files'))

    # Argument checking
    los, lats, lons, heights, flag, weather_model, wmLoc, zref, outformat, \
         time, out, download_only, parallel, verbose = checkArgs(args, p)

    if verbose: 
       print('Starting to run the weather model calculation')
       print('Time type: {}'.format(type(time)))
       print('Time: {}'.format(time.strftime('%Y%m%d')))
       print('Parallel is {}'.format(parallel))

    wetDelay, hydroDelay = \
       delay.tropo_delay(time, los, lats, lons, heights, 
                         weather_model, wmLoc, zref, out,
                         parallel=parallel, verbose = verbose, 
                         download_only = download_only)

    writeDelays(flag, wetDelay, hydroDelay, time, los,
                out, outformat, weather_model['name'],
                proj = None, gt = None)

