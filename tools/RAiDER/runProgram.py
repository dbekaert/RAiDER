import os
import numpy as np

import RAiDER.util
from RAiDER.util import parse_date, parse_time
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
    p = argparse.ArgumentParser(
        description='Calculate tropospheric delay from a weather model')

    p.add_argument(
        '--date',dest='dateList',
        help='Fetch weather model data for a given date or date range.\nCan be a single date or a comma-separated list of two dates (earlier, later) in the ISO 8601 format',
        type=read_date, required=True)
    p.add_argument(
        '--time', dest = 'time',
        help='Fetch weather model data at this (ISO 8601 format) time of day',
        type=parse_time, required=True)

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
        type=int, default=_ZREF)

    p.add_argument(
        '--outformat', help='Output file format; GDAL-compatible for DEM, HDF5 for height levels',
        default=None)

    p.add_argument('--out', help='Output file directory', default='.')

    p.add_argument('--no_parallel', '-p', action='store_true',dest='no_parallel', default = False, help='Do not run operation in parallel? Default False. Recommend only True for verbose (debug) mode')

    p.add_argument('--download_only', action='store_true',dest='download_only', default = False, help='Download weather model only without processing? Default False')

    p.add_argument('--verbose', '-v', action='store_true',dest='verbose', default = False, help='Run in verbose (debug) mode? Default False')

    return p.parse_args(), p


def writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                outformat, wetFilename, hydroFilename, 
                proj = None, gt = None, ndv = 0.):
    '''
    Write the delay numpy arrays to files in the format specified
    '''

    # Need to consistently handle noDataValues
    wetDelay[np.isnan(wetDelay)] = ndv
    hydroDelay[np.isnan(hydroDelay)] = ndv
   
    # Do different things, depending on the type of input
    if flag=='station_file':
        import pandas as pd
        df = pd.read_csv(wetFilename)

        # quick check for consistency
        assert(np.all(np.abs(lats - df['Lat']) < 0.01))

        df['wetDelay'] = wetDelay
        df['hydroDelay'] = hydroDelay
        df['totalDelay'] = wetDelay + hydroDelay
        df.to_csv(wetFilename, index=False)

    elif flag=='netcdf':
        RAiDER.util.writeResultsToNETCDF(lats, lons, wetDelay, wetFilename, noDataValue = ndv,
                       fmt=outformat, proj=proj, gt=gt)
        RAiDER.util.writeResultsToNETCDF(lats, lons, hydroDelay, hydroFilename, noDataValue = ndv,
                       fmt=outformat, proj=proj, gt=gt)

    else:
        RAiDER.util.writeArrayToRaster(wetDelay, wetFilename, noDataValue = ndv,
                       fmt = outformat, proj = proj, gt = gt)
        RAiDER.util.writeArrayToRaster(hydroDelay, hydroFilename, noDataValue = ndv,
                       fmt = outformat, proj = proj, gt = gt)


def main(los, lats, lons, heights, flag, weather_model, wmLoc, zref, 
         outformat, time, out, download_only, parallel, verbose, 
         wetFilename, hydroFilename):
    """
    raiderDelay main function.
    """
    from RAiDER.delay import tropo_delay

    if verbose: 
       print('Starting to run the weather model calculation')
       print('Time type: {}'.format(type(time)))
       print('Time: {}'.format(time.strftime('%Y%m%d')))
       print('Parallel is {}'.format(parallel))

    wetDelay, hydroDelay = \
       tropo_delay(time, los, lats, lons, heights, 
                         weather_model, wmLoc, zref, out,
                         parallel=parallel, verbose = verbose, 
                         download_only = download_only)

    writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                outformat, wetFilename, hydroFilename,
                proj = None, gt = None, ndv = 0.)


def parseCMD():
    """
    Parse command-line arguments and pass to main
    We'll parse arguments and call delay.py.
    """
    from RAiDER.checkArgs import checkArgs
    from RAiDER.llreader import getHeights 

    args, p = parse_args()

    # Argument checking
    los, lats, lons, heights, flag, weather_model, wmLoc, zref, outformat, \
         times, out, download_only, parallel, verbose, \
         wetNames, hydroNames= checkArgs(args, p)

    # Pull the DEM
    if verbose: 
        print('Beginning DEM calculation')
    lats, lons, hgts = getHeights(lats, lons,heights)
    import pdb; pdb.set_trace()

    # Loop over each datetime and compute the delay
    for t, wfn, hfn in zip(times, wetNames, hydroNames):
        main(los, lats, lons, hgts, flag, weather_model, wmLoc, zref,
             outformat, t, out, download_only, parallel, verbose, wfn, hfn)

