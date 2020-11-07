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
import multiprocessing
import os

import pandas as pd
import requests

from RAiDER.cli.parser import add_cpus, add_out, add_verbose
from RAiDER.logger import *
from RAiDER.getStationDelays import get_station_data

# base URL for UNR repository
_UNR_URL = "http://geodesy.unr.edu/"


def create_parser():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Check for and download tropospheric zenith delays for a set of GNSS stations from UNR

Example call to virtually access and append zenith delay information to a CSV table in specified output
directory, across specified range of years and all available times of day, and confined to specified
geographic bounding box :
downloadGNSSdelay.py --out products -y '2010,2014' -b '39 40 -79 -78'

Example call to virtually access and append zenith delay information to a CSV table in specified output
directory, across specified range of years and specified time of day, and distributed globally :
downloadGNSSdelay.py --out products -y '2010,2014' --returntime '00:00:00' -f station_list.txt

Example call to virtually access and append zenith delay information to a CSV table in specified output
directory, across specified range of years and specified time of day, and distributed globally but restricted
to list of stations specified in input textfile :
downloadGNSSdelay.py --out products -y '2010,2014' --returntime '00:00:00' -f station_list.txt

NOTE, following example call to physically download zenith delay information not recommended as it is not
necessary for most applications.
Example call to physically download and append zenith delay information to a CSV table in specified output
directory, across specified range of years and specified time of day, and confined to specified
geographic bounding box :
downloadGNSSdelay.py --download --out products -y '2010,2014' --returntime '00:00:00' -b '39 40 -79 -78'
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
        '--years', '-y', dest='years',
        help="""Year to check or download delays (format YYYY).
Can be a single value or a comma-separated list. If two years non-consecutive years are given, download each year in between as well.
""", type=parse_years, required=True)

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

    return p


def cmd_line_parse(iargs=None):
    parser = create_parser()
    return parser.parse_args(args=iargs)


def get_station_list(bbox=None, writeLoc=None, userstatList=None, name_appendix=''):
    '''
    Creates a list of stations inside a lat/lon bounding box from a source
    Inputs:
        bbox    - length-4 list of floats that describes a bounding box. Format is
                  S N W E
    '''
    writeLoc = os.path.join(writeLoc or os.getcwd(), 'gnssStationList_overbbox' + name_appendix + '.csv')

    if userstatList:
        userstatList = read_text_file(userstatList)

    statList = get_stats_by_llh(llhBox=bbox, userstatList=userstatList)

    # write to file and pass final stations list
    statList.to_csv(writeLoc, index=False)
    stations = list(statList['ID'].values)

    return stations, writeLoc


def get_stats_by_llh(llhBox=None, baseURL=_UNR_URL, userstatList=None):
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
    for ind, line in enumerate(data.text.splitlines()):  # files are iterable
        if ind == 0:
            continue
        statID, lat, lon, height = get_ID(line)
        # Only pass if in bbox
        # And if user list of stations specified, only pass info for stations within list
        if in_box(lat, lon, llhBox) and (not userstatList or statID in userstatList):
            # convert lon into range [-180,180]
            lon = fix_lons(lon)
            stations.append({'ID': statID, 'Lat': lat, 'Lon': lon, 'Hgt_m': height})

    logger.info('%d stations were found', len(stations))
    stations = pd.DataFrame(stations)
    # Report stations from user's list that do not cover bbox
    if userstatList:
        userstatList = [
            i for i in userstatList if i not in stations['ID'].to_list()]
        if userstatList:
            logger.warning(
                "The following user-input stations are not covered by the input "
                "bounding box %s: %s",
                str(llhBox).strip('[]'), str(userstatList).strip('[]')
            )

    return stations


def download_tropo_delays(stats, years, gps_repo=None, writeDir='.', numCPUs=8, download=False):
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
    stat_year_tup = itertools.product(stats, years)
    stat_year_tup = ((*tup, writeDir, download) for tup in stat_year_tup)
    # Parallelize remote querying of station locations
    with multiprocessing.Pool(numCPUs) as multipool:
        # only record valid path
        if gps_repo == 'UNR':
            results = [
                fileurl for fileurl in multipool.starmap(download_UNR, stat_year_tup)
                if fileurl['path']
            ]

    # Write results to file
    statDF = pd.DataFrame(results).set_index('ID')
    statDF.to_csv(os.path.join(writeDir, '{}gnssStationList_overbbox_withpaths.csv'.format(gps_repo)))


def download_UNR(statID, year, writeDir='.', download=False, baseURL=_UNR_URL):
    '''
    Download a zip file containing tropospheric delays for a given station and year
    The URL format is http://geodesy.unr.edu/gps_timeseries/trop/<ssss>/<ssss>.<yyyy>.trop.zip
    Inputs:
        statID   - 4-character station identifier
        year     - 4-numeral year
    '''
    URL = "{0}gps_timeseries/trop/{1}/{1}.{2}.trop.zip".format(
        baseURL, statID.upper(), year)
    logger.debug('Currently checking station %s in %s', statID, year)
    if download:
        saveLoc = os.path.abspath(os.path.join(
            writeDir, '{0}.{1}.trop.zip'.format(statID.upper(), year)))
        filepath = download_url(URL, saveLoc)
    else:
        filepath = check_url(URL)
    return {'ID': statID, 'year': year, 'path': filepath}


def download_url(url, save_path, chunk_size=2048):
    '''
    Download a file from a URL. Modified from
    https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
    '''
    r = requests.get(url, stream=True)
    if r.status_code == 404:
        return ''
    else:
        logger.debug('Beginning download of %s to %s', url, save_path)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
        logger.debug('Completed download of %s to %s', url, save_path)
        return save_path


def check_url(url):
    '''
    Check whether a file exists at a URL. Modified from
    https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
    '''
    r = requests.head(url)
    if r.status_code == 404:
        url = ''
    return url


def read_text_file(filename):
    '''
    Read a list of GNSS station names from a plain text file
    '''
    with open(filename, 'r') as f:
        return [line.strip() for line in f]


def in_box(lat, lon, llhbox):
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


def get_ID(line):
    '''
    Pulls the station ID, lat, lon, and height for a given entry in the UNR text file
    '''
    stat_id, lat, lon, height = line.split()[:4]
    return stat_id, float(lat), float(lon), float(height)


def parse_years(timestr):
    '''
    Takes string input and returns a list of years as integers
    '''
    years = list(map(int, timestr.split(',')))
    # If two years non-consecutive years are given, query for each year in between
    if len(years) == 2:
        years = list(range(years[0], years[1] + 1))
    return years


def query_repos(
    station_file,
    bounding_box,
    gps_repo,
    out,
    years,
    returnTime,
    download,
    cpus,
    verbose
):
    """
    Main workflow for querying supported GPS repositories for zenith delay information.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Create specified output directory if it does not exist.
    if not os.path.exists(out):
        os.mkdir(out)

    # Setup bounding box
    if bounding_box:
        if not os.path.isfile(bounding_box):
            try:
                bbox = [float(val) for val in bounding_box.split()]
            except ValueError:
                raise Exception(
                    'Cannot understand the --bbox argument. String input is incorrect or path does not exist.')
            if bbox[2] * bbox[3] < 0:
                long_cross_zero = 1
            else:
                long_cross_zero = 0
            # if necessary, convert negative longitudes to positive
            if bbox[2] < 0:
                bbox[2] += 360
            if bbox[3] < 0:
                bbox[3] += 360
    # If bbox not specified, query stations across the entire globe
    else:
        bbox = [-90, 90, 0, 360]

    # Handle station query
    if long_cross_zero is 1:
        bbox1 = bbox.copy()
        bbox2 = bbox.copy()
        bbox1[3] = 360.0
        bbox2[2] = 0.0
        stats1, origstatsFile1 = get_station_list(bbox=bbox1, writeLoc=out, userstatList=station_file, name_appendix='_a')
        stats2, origstatsFile2 = get_station_list(bbox=bbox2, writeLoc=out, userstatList=station_file, name_appendix='_b')
        stats = stats1 + stats2
        origstatsFile = origstatsFile1[:-6] + '.csv'
        file_a = pd.read_csv(origstatsFile1)
        file_b = pd.read_csv(origstatsFile2)
        frames = [file_a, file_b]
        result = pd.concat(frames, ignore_index=True)
        result.to_csv(origstatsFile, index=False)
    else:
        if bbox[3] < bbox[2]:
            bbox[3] = 360.0
        stats, origstatsFile = get_station_list(
            bbox=bbox, writeLoc=out, userstatList=station_file)

    # iterate over years
    download_tropo_delays(
        stats, years, gps_repo=gps_repo, writeDir=out, download=download
    )

    # Add lat/lon info
    origstatsFile = pd.read_csv(origstatsFile)
    statsFile = pd.read_csv(os.path.join(
        out, '{}gnssStationList_overbbox_withpaths.csv'.format(gps_repo)))
    statsFile = pd.merge(left=statsFile, right=origstatsFile,
                         how='left', left_on='ID', right_on='ID')
    statsFile.to_csv(os.path.join(
        out, '{}gnssStationList_overbbox_withpaths.csv'.format(gps_repo)), index=False)
    del origstatsFile, statsFile

    # Extract delays for each station
    get_station_data(
        os.path.join(out, '{}gnssStationList_overbbox_withpaths.csv'.format(gps_repo)),
        gps_repo=gps_repo,
        numCPUs=cpus,
        outDir=out,
        returnTime=returnTime
    )

    logger.debug('Completed processing')


if __name__ == "__main__":
    inps = cmd_line_parse()

    query_repos(
        inps.station_file,
        inps.bounding_box,
        inps.gps_repo,
        inps.out,
        inps.years,
        inps.returnTime,
        inps.download,
        inps.cpus,
        inps.verbose
    )
