# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Simran Sangha, & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import itertools
import multiprocessing as mp
import os

import pandas as pd

from RAiDER.getStationDelays import get_station_data
from RAiDER.logger import logger, logging
from RAiDER.models.customExceptions import NoStationDataFoundError
from RAiDER.utilFcns import requests_retry_session


# base URL for UNR repository
_UNR_URL = 'https://geodesy.unr.edu/'
NEW_STATION_FILENAME = 'gnssStationList_overbbox'


def get_station_list(
    bbox=None,
    stationFile=None,
    writeLoc=None,
    name_appendix='',
    writeStationFile=True,
):
    """
    Creates a list of stations inside a lat/lon bounding box from a source.

    Args:
        bbox: list of float     - length-4 list of floats that describes a bounding box.
                                  Format is S N W E
        station_file: str       - Name of a .csv or .txt file to read containing station IDs
        writeStationFile: bool  - Whether to write out the station dataframe to a .csv file
        writeLoc: string    - Directory to write data
        name_appendix: str  - name to append to output file

    Returns:
        stations: list of strings        - station IDs to access
        output_file: string or dataframe - file to write delays
    """
    if stationFile is not None:
        try:
            station_data = pd.read_csv(stationFile)
        except:
            stations = []
            with open(stationFile) as f:
                for k, line in enumerate(f):
                    if k == 0:
                        names = line.strip().split()
                    else:
                        stations.append(line.strip().split())
            station_data = pd.DataFrame(stations, columns=names)         
    else:
        station_data = get_stats_by_llh(llhBox=bbox)

    # write to file and pass final stations list
    if writeStationFile:
        output_file = os.path.join(writeLoc or os.getcwd(), NEW_STATION_FILENAME + name_appendix + '.csv')
        station_data.to_csv(output_file, index=False)

    return list(station_data['ID'].values), [output_file if writeStationFile else station_data][0]


def get_stats_by_llh(llhBox=None, baseURL=_UNR_URL):
    """
    Function to pull lat, lon, height, beginning date, end date, and number of solutions for stations inside the bounding box llhBox.
    llhBox should be a tuple in SNWE format.
    """
    if llhBox is None:
        llhBox = [-90, 90, 0, 360]
    S, N, W, E = llhBox
    if (W < 0) or (E < 0):
        raise ValueError('get_stats_by_llh: bounding box must be on lon range [0, 360]')

    stationHoldings = f'{baseURL}NGLStationPages/llh.out'
    # it's a file like object and works just like a file

    stations = pd.read_csv(stationHoldings, sep=r'\s+', names=['ID', 'Lat', 'Lon', 'Hgt_m'])

    # convert lons from [0, 360] to [-180, 180]
    stations['Lon'] = ((stations['Lon'].values + 180) % 360) - 180

    stations = filterToBBox(stations, llhBox)

    return stations


def download_tropo_delays(
    stats,
    years,
    gps_repo='UNR',
    writeDir='.',
    numCPUs=8,
    download=False,
) -> None:
    """
    Check for and download GNSS tropospheric delays from an archive. If
    download is True then files will be physically downloaded, but this
    is not necessary as data can be virtually accessed.

    Args:
        stats: stations     - Stations to access
        years: list of int  - A list of years to be downloaded
        gps_repo: string    - SNWE bounds target area to ensure weather model contains them
        writeDir: string    - False if preprocessing weather model data
        numCPUs: int        - whether to write debug plots
        download: bool      - True if you want to download even when the weather model exists

    Returns:
        None
    """
    # argument checking
    if not isinstance(stats, (list, str)):
        raise TypeError('stats should be a string or a list of strings')
    if not isinstance(years, (list, int)):
        raise TypeError('years should be an int or a list of ints')

    # Iterate over stations and years and check or download data
    stat_year_tup = itertools.product(stats, years)
    stat_year_tup = ((*tup, writeDir, download) for tup in stat_year_tup)

    # Parallelize remote querying of station locations
    results = []
    with mp.Pool(numCPUs) as multipool:
        # only record valid path
        if gps_repo == 'UNR':
            results = [fileurl for fileurl in multipool.starmap(download_UNR, stat_year_tup) if fileurl['path']]
        else:
            raise NotImplementedError(f'download_tropo_delays: gps_repo "{gps_repo}" not yet implemented')

    # Write results to file
    if len(results) == 0:
        raise NoStationDataFoundError(station_list=stats, years=years)
    statDF = pd.DataFrame(results).set_index('ID')
    statDF.to_csv(os.path.join(writeDir, f'{gps_repo}{NEW_STATION_FILENAME}_withpaths.csv'))


def download_UNR(statID, year, writeDir='.', download=False, baseURL=_UNR_URL):
    """
    Download a zip file containing tropospheric delays for a given station and year.
    The URL format is http://geodesy.unr.edu/gps_timeseries/trop/<ssss>/<ssss>.<yyyy>.trop.zip
    Inputs:
        statID   - 4-character station identifier
        year     - 4-numeral year
    """
    if baseURL not in [_UNR_URL]:
        raise NotImplementedError(f'Data repository {baseURL} has not yet been implemented')

    URL = '{0}gps_timeseries/trop/{1}/{1}.{2}.trop.zip'.format(baseURL, statID.upper(), year)
    logger.debug('Currently checking station %s in %s', statID, year)
    if download:
        saveLoc = os.path.abspath(os.path.join(writeDir, f'{statID.upper()}.{year}.trop.zip'))
        filepath = download_url(URL, saveLoc)
        if filepath == '':
            raise ValueError('Year or station ID does not exist')
    else:
        filepath = check_url(URL)
    return {'ID': statID, 'year': year, 'path': filepath}


def download_url(url, save_path, chunk_size=2048):
    """
    Download a file from a URL. Modified from
    https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url.
    """
    session = requests_retry_session()
    r = session.get(url, stream=True)

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
    """
    Check whether a file exists at a URL. Modified from
    https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url.
    """
    session = requests_retry_session()
    r = session.head(url)
    if r.status_code == 404:
        url = ''
    return url


def in_box(lat, lon, llhbox):
    """Checks whether the given lat, lon pair are inside the bounding box llhbox."""
    return lat < llhbox[1] and lat > llhbox[0] and lon < llhbox[3] and lon > llhbox[2]


def fix_lons(lon):
    """Fix the given longitudes into the range `[-180, 180]`."""
    fixed_lon = ((lon + 180) % 360) - 180
    # Make the positive 180s positive again.
    if fixed_lon == -180 and lon > 0:
        fixed_lon *= -1
    return fixed_lon


def get_ID(line):
    """Pulls the station ID, lat, lon, and height for a given entry in the UNR text file."""
    stat_id, lat, lon, height = line.split()[:4]
    return stat_id, float(lat), float(lon), float(height)


def main(inps=None) -> None:
    """Main workflow for querying supported GPS repositories for zenith delay information."""
    try:
        dateList = inps.date_list
        returnTime = inps.time
    except:
        dateList = inps.dateList
        returnTime = inps.returnTime

    station_file = inps.station_file
    if (station_file is not None) and not os.path.isfile(station_file):
        raise FileNotFoundError(f'File {station_file} does not exist.')
    bounding_box = inps.bounding_box
    gps_repo = inps.gps_repo
    out = inps.out
    download = inps.download
    cpus = inps.cpus
    verbose = inps.verbose

    if verbose:
        logger.setLevel(logging.DEBUG)

    # Create specified output directory if it does not exist.
    if not os.path.exists(out):
        os.mkdir(out)

    # Setup bounding box
    if bounding_box:
        bbox, long_cross_zero = parse_bbox(bounding_box)
    # If bbox not specified, query stations across the entire globe
    else:
        bbox = [-90, 90, 0, 360]
        long_cross_zero = 1

    # Handle station query
    stats, statdf = get_stats(bbox, long_cross_zero, out, station_file)

    # iterate over years
    years = list(set([i.year for i in dateList]))
    download_tropo_delays(stats, years, gps_repo=gps_repo, writeDir=out, download=download)

    # Combine station data with URL info
    pathsdf = pd.read_csv(os.path.join(out, f'{gps_repo}{NEW_STATION_FILENAME}_withpaths.csv'))
    pathsdf = pd.merge(left=pathsdf, right=statdf, how='left', left_on='ID', right_on='ID')
    pathsdf.to_csv(os.path.join(out, f'{gps_repo}{NEW_STATION_FILENAME}_withpaths.csv'), index=False)
    del statdf, pathsdf

    # Extract delays for each station
    dateList = [k.strftime('%Y-%m-%d') for k in dateList]
    get_station_data(
        os.path.join(out, f'{gps_repo}{NEW_STATION_FILENAME}_withpaths.csv'),
        dateList,
        gps_repo=gps_repo,
        numCPUs=cpus,
        outDir=out,
        returnTime=returnTime,
    )

    logger.debug('Completed processing')


def parse_bbox(bounding_box):
    """Parse bounding box arguments."""
    if isinstance(bounding_box, str) and not os.path.isfile(bounding_box):
        try:
            bbox = [float(val) for val in bounding_box.split()]
        except ValueError:
            raise Exception('Cannot understand the --bbox argument. String input is incorrect or path does not exist.')
    elif isinstance(bounding_box, list):
        bbox = bounding_box

    else:
        raise Exception('Passing a file with a bounding box not yet supported.')

    long_cross_zero = 1 if bbox[2] * bbox[3] < 0 else 0

    # if necessary, convert negative longitudes to positive
    if bbox[2] < 0:
        bbox[2] += 360

    if bbox[3] < 0:
        bbox[3] += 360

    return bbox, long_cross_zero


def get_stats(bbox, long_cross_zero, out, station_file):
    """Pull the stations needed."""
    if long_cross_zero == 1:
        bbox1 = bbox.copy()
        bbox2 = bbox.copy()
        bbox1[3] = 360.0
        bbox2[2] = 0.0
        stats1, statdata1 = get_station_list(
            bbox=bbox1, stationFile=station_file, name_appendix='_a', writeStationFile=False
        )
        stats2, statdata2 = get_station_list(
            bbox=bbox2, stationFile=station_file, name_appendix='_b', writeStationFile=False
        )
        stats = stats1 + stats2
        stats = list(set(stats))
        frames = [statdata1, statdata2]
        statdata = pd.concat(frames, ignore_index=True)
        statdata = statdata.drop_duplicates(subset=['ID'])
    else:
        if bbox[3] < bbox[2]:
            bbox[3] = 360.0
        stats, statdata = get_station_list(bbox=bbox, stationFile=station_file, writeStationFile=False)

    statdata.to_csv(NEW_STATION_FILENAME + '.csv', index=False)
    return stats, statdata


def filterToBBox(stations, llhBox):
    """
    Filter a dataframe by lat/lon.
    *NOTE: llhBox longitude format should be [0, 360].

    Args:
        stations: DataFrame     - a pandas dataframe with "Lat" and "Lon" columns
        llhBox: list of float   - 4-element list: [S, N, W, E]

    Returns:
        a Pandas Dataframe with stations removed that are not inside llhBox
    """
    S, N, W, E = llhBox
    if (W < 0) or (E < 0):
        raise ValueError('llhBox longitude format should 0-360')

    # For a user-provided file, need to check the column names
    keys = stations.columns
    lat_keys = ['lat', 'latitude', 'Lat', 'Latitude']
    lon_keys = ['lon', 'longitude', 'Lon', 'Longitude']
    index = None
    for k, key in enumerate(lat_keys):
        if key in list(keys):
            index = k
            break
    if index is None:
        raise KeyError('filterToBBox: No valid column names found for latitude and longitude')
    lon_key = lon_keys[k]
    lat_key = lat_keys[k]

    if stations[lon_key].min() < 0:
        # convert lon format to -180 to 180
        W, E = (((D + 180) % 360) - 180 for D in [W, E])

    mask = (stations[lat_key] > S) & (stations[lat_key] < N) & (stations[lon_key] < E) & (stations[lon_key] > W)
    return stations[mask]
