#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Simran Sangha, & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import argparse
import itertools
import os
import sys

import pandas as pd
import requests
from RAiDER.getStationDelays import getStationData

# base URL for UNR repository
_UNR_URL = "http://geodesy.unr.edu/"

def parse_args():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Check for and download tropospheric zenith delays for a set of GNSS stations from UNR
Usage examples: 
downloadGNSSdelay.py -y 2010 --check
downloadGNSSdelay.py -y 2010 -b 40 -79 39 -78 -v
downloadGNSSdelay.py -y 2010 -f station_list.txt --out products
""")

    p.add_argument(
        '--years', '-y', dest='years', nargs='+',
        help="""Year to check or download delays (format YYYY).
Can be a single value or a comma-separated list. If two years non-consecutive years are given, download each year in between as well. 
""", type=parse_years, required=True)

    # Stations to check/download
    area = p.add_argument_group(
        'Stations to check/download. Can be a lat/lon bounding box or file, or will run the whole world if not specified')
    area.add_argument(
        '--station_file', '-f', default=None, dest='station_file',
        help=('Text file containing a list of 4-char station IDs separated by newlines'))
    area.add_argument(
        '--BBOX', '-b', dest='bounding_box', type=str, default=None,
        help="Provide either valid shapefile or Lat/Lon Bounding SNWE. -- Example : '19 20 -99.5 -98.5'")

    misc = p.add_argument_group("Run parameters")
    misc.add_argument(
        '--outformat',
        help='GDAL-compatible file format if surface delays are requested.',
        default=None)

    misc.add_argument(
        '--out', dest='out',
        help='Directory to download products',
        default='.')

    misc.add_argument(
        '--returntime', dest='returnTime',
        help="Return delays closest to this specified time. If not specified, the GPS delays for all times will be returned. Input in 'HH:MM:SS', e.g. '16:00:00'",
        default=None)

    misc.add_argument(
        '--download',
        help='Physically download data. Note this is not necessary to proceed with statistical analysis, as data can be handled virtually in the program.',
        action='store_true', dest='download', default=False)

    misc.add_argument(
        '--verbose', '-v',
        help='Run in verbose (debug) mode? Default False',
        action='store_true', dest='verbose', default=False)

    return p.parse_args(), p


def parseCMD():
    """
    Parse command-line arguments and pass to tropo_delay
    We'll parse arguments and call delay.py.
    """
    args, p = parse_args()

    # Create specified output directory if it does not exist.
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    # Setup bounding box
    if args.bounding_box:
        if isinstance([str(val) for val in args.bounding_box.split()], list) and not os.path.isfile(args.bounding_box):
            try:
                bbox = [float(val) for val in args.bounding_box.split()]
            except:
                raise Exception(
                    'Cannot understand the --bbox argument. String input is incorrect or path does not exist.')
            # if necessary, convert negative longitudes to positive
            if bbox[2] < 0:
                bbox[2] += 360
            if bbox[3] < 0:
                bbox[3] += 360
    # If bbox not specified, query stations across the entire globe
    else:
        bbox = [-90, 90, 0, 360]

    # Handle station query
    stats, origstatsFile = getStationList(
        bbox=bbox, writeLoc=args.out, userstatList=args.station_file)

    # iterate over years
    for yr in args.years:
        statDF = downloadTropoDelays(
            stats, yr, writeDir=args.out, download=args.download, verbose=args.verbose)
        statDF.to_csv(os.path.join(
            args.out, 'gnssStationList_overbbox_withpaths.csv'))

    # Add lat/lon info
    origstatsFile = pd.read_csv(origstatsFile)
    statsFile = pd.read_csv(os.path.join(
        args.out, 'gnssStationList_overbbox_withpaths.csv'))
    statsFile = pd.merge(left=statsFile, right=origstatsFile,
                         how='left', left_on='ID', right_on='ID')
    statsFile.to_csv(os.path.join(
        args.out, 'gnssStationList_overbbox_withpaths.csv'), index=False)
    del statDF, origstatsFile, statsFile

    # Extract delays for each station
    getStationData(os.path.join(args.out, 'gnssStationList_overbbox_withpaths.csv'),
                   outDir=args.out, returnTime=args.returnTime)

    if args.verbose:
        print('Completed processing')


def getStationList(bbox=None, writeLoc=None, userstatList=None):
    '''
    Creates a list of stations inside a lat/lon bounding box from a source
    Inputs: 
        bbox    - length-4 list of floats that describes a bounding box. Format is 
                  S N W E
    '''
    writeLoc = os.path.join(writeLoc or os.getcwd(), 'gnssStationList_overbbox.csv')

    if userstatList:
        userstatList = readTextFile(userstatList)

    statList = getStatsByllh(llhBox=bbox, userstatList=userstatList)

    # write to file and pass final stations list
    statList.to_csv(writeLoc, index=False)
    stations = list(statList['ID'].values)

    return stations, writeLoc


def getStatsByllh(llhBox=None, baseURL=_UNR_URL, userstatList=None):
    '''
    Function to pull lat, lon, height, beginning date, end date, and number of solutions for stations inside the bounding box llhBox. 
    llhBox should be a tuple with format (lat1, lat2, lon1, lon2), where lat1, lon1 define the lower left-hand corner and lat2, lon2 
    define the upper right corner. 
    '''
    if llhBox is None:
        llhBox = [-90, 90, 0, 360]

    stationHoldings = '{}NGLStationPages/DataHoldings.txt'.format(baseURL)
    # it's a file like object and works just like a file
    data = requests.get(stationHoldings)
    stations = []
    for ind, line in enumerate(data):  # files are iterable
        if ind == 0:
            continue
        statID, lat, lon = getID(line)
        # Only pass if in bbox
        # And if user list of stations specified, only pass info for stations within list
        if inBox(lat, lon, llhBox) and (not userstatList or statID in userstatList):
            # convert lon into range [-180,180]
            lon = fix_lons(lon)
            stations.append({'ID': statID, 'Lat': lat, 'Lon': lon})

    print('{} stations were found'.format(len(stations)))
    stations = pd.DataFrame(stations)
    # Report stations from user's list that do not cover bbox
    if userstatList:
        userstatList = [
            i for i in userstatList if i not in stations['ID'].to_list()]
        if userstatList:
            print("Warning: The following user-input stations are not covered by the input bounding box {0}: {1}"
                  .format(str(llhBox).strip('[]'), str(userstatList).strip('[]')))

    return stations


def downloadTropoDelays(stats, years, writeDir='.', download=False, verbose=False):
    '''
    Check for and download GNSS tropospheric delays from an archive. If download is True then 
    files will be physically downloaded, which again is not necessary as data can be virtually accessed. 
    '''

    # argument checking
    if not isinstance(stats, (list, str)):
        raise TypeError('stats should be a string or a list of strings')
    if not isinstance(years, (list, int)):
        raise TypeError('years should be an int or a list of ints')

    # Iterate over stations and years and check or download data
    results = []
    stat_year_tup = itertools.product(stats, years)
    for s, y in itertools.product(stats, years):
        if verbose:
            print('Currently checking station {} in {}'.format(s, y))
        fileurl = download_UNR(s, y, writeDir=writeDir,
                               download=download, verbose=verbose)
        # only record valid path
        if fileurl:
            results.append({'ID': s, 'year': y, 'path': fileurl})
    return pd.DataFrame(results).set_index('ID')


def download_UNR(statID, year, writeDir='.', baseURL=_UNR_URL, download=False, verbose=False):
    '''
    Download a zip file containing tropospheric delays for a given station and year
    The URL format is http://geodesy.unr.edu/gps_timeseries/trop/<ssss>/<ssss>.<yyyy>.trop.zip
    Inputs:
        statID   - 4-character station identifier
        year     - 4-numeral year
    '''
    URL = "{0}gps_timeseries/trop/{1}/{1}.{2}.trop.zip".format(
        baseURL, statID.upper(), year)
    if download:
        saveLoc = os.path.abspath(os.path.join(
            writeDir, '{0}.{1}.trop.zip'.format(statID.upper(), year)))
        filepath = download_url(URL, saveLoc, verbose=verbose)
    else:
        filepath = check_url(URL, verbose=verbose)
    return filepath


def download_url(url, save_path, verbose=False, chunk_size=2048):
    '''
    Download a file from a URL. Modified from 
    https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
    '''
    r = requests.get(url, stream=True)
    if r.status_code == 404:
        return ''
    else:
        if verbose:
            print('Beginning download of {} to {}'.format(url, save_path))
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
        if verbose:
            print('Completed download of {} to {}'.format(url, save_path))
        return save_path


def check_url(url, verbose=False):
    '''
    Check whether a file exists at a URL. Modified from 
    https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
    '''
    r = requests.head(url)
    if r.status_code == 404:
        url = ''
    return url


def readTextFile(filename):
    '''
    Read a list of GNSS station names from a plain text file
    '''
    with open(filename, 'r') as f:
        return [line.strip() for line in f]


def inBox(lat, lon, llhbox):
    '''
    Checks whether the given lat, lon pair are inside the bounding box llhbox
    '''
    return lat < llhbox[1] and lat > llhbox[0] and lon < llhbox[3] and lon > llhbox[2]


def fix_lons(lon):
    """
    Fix the given longitudes into the range ``[-180, 180]``.
    """
    fixed_lon = ((lon + 180) % 360) - 180
    # Make the positive 180s positive again.
    if fixed_lon == -180 and lon > 0:
        fixed_lon *= -1
    return fixed_lon


def getID(line):
    '''
    Pulls the station ID, lat, and lon for a given entry in the UNR text file
    '''
    stat_id, lat, lon = line.decode().split()[:3]
    return stat_id, float(lat), float(lon)


def parse_years(timestr):
    '''
    Takes string input and returns a list of years as integers
    '''
    years = list(map(int, timestr.split(',')))
    # If two years non-consecutive years are given, query for each year in between
    if len(years) == 2:
         years = list(range(years[0],years[1]+1))
    return years
