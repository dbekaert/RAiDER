# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Simran Sangha, & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import datetime as dt
import gzip
import io
import multiprocessing as mp
import os
import zipfile

from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import requests

from RAiDER.logger import logger


def get_delays_UNR(stationFile: Path, filename: str, dateList: List, returnTime: str=None) -> None:
    """
    Parses and returns a dictionary containing either (1) all
    the GPS delays, if returnTime is None, or (2) only the delay
    at the closest times to to returnTime.

    Args:
         stationFile: binary        - a .gz station delay file
         filename: ?                - ?
         dateList: list of datetime - ?
         returnTime: datetime       - specified time of GPS delay (default all times)

    Returns:
        None

    The function writes a CSV file containing the times and delay information
    (delay in mm, delay uncertainty, delay gradients)

    Refer to the following sites to interpret stationFile variable names:
    ftp://igs.org/pub/data/format/sinex_tropo.txt
    http://geodesy.unr.edu/gps_timeseries/README_trop2.txt
    Wet and hydrostratic delays were derived as so:
    Constants —> k1 = 0.704, k2 = 0.776, k3 = 3739.0, m = 18.0152/28.9644,
    k2' = k2-(k1*m) = 0.33812796398337275, Rv = 461.5 J/(kg·K), ρl = 997 kg/m^3

    *NOTE: wet delays passed here are computed using
    PMV = precipitable water vapor,
    P = total atm pressure,
    Tm = mean temp of the column, as:

        Wet zenith delay = 10^-6 ρlRv(k2' + k3/Tm) PMV
        Hydrostatic zenith delay = Total zenith delay - wet zenith delay = k1*(P/Tm)

    Source —> Hanssen, R. F. (2001) eqns. 6.2.7-10

    *NOTE: Due to a formatting error in the tropo SINEX files, the two
    tropospheric gradient columns (TGNTOT and TGETOT) are interchanged,
    as are the formal error columns (_SIG).

    Source  —> http://geodesy.unr.edu/gps_timeseries/README_trop2.txt)
    """
    # sort through station zip files
    allstationTarfiles = []
    # if URL
    if stationFile.startswith('http'):
        r = requests.get(stationFile)
        ziprepo = zipfile.ZipFile(io.BytesIO(r.content))
    else:
        ziprepo = zipfile.ZipFile(stationFile)
    # iterate through tarfiles
    stationTarlist = sorted(ziprepo.namelist())

    final_stationTarlist = []
    for j in stationTarlist:
        # get the date of the file
        time, _, doyFromFile = get_date(Path.name(j).split('.'))  # noqa: PTH119
        # check if in list of specified input dates
        if time.strftime('%Y-%m-%d') not in dateList:
            continue
        final_stationTarlist.append(j)
        f = gzip.open(ziprepo.open(j), 'rb')
        # initialize variables
        d, Sig, dwet, dhydro, timesList = [], [], [], [], []
        flag = False
        for line in f.readlines():
            try:
                line = line.decode('utf-8')
            except UnicodeDecodeError:
                line = line.decode('latin-1')
            if flag:
                # Do not attempt to read header
                if 'SITE' in line:
                    continue
                # Attempt to read data
                try:
                    split_lines = line.split()
                    # units: mm, mm, mm, deg, deg, deg, deg, mm, mm, K
                    trotot, trototSD, trwet, tgetot, tgetotSD, tgntot, tgntotSD, wvapor, wvaporSD, mtemp = (
                        float(t) for t in split_lines[2:]
                    )
                except:  # TODO: What error(s)?
                    continue
                site = split_lines[0]
                year, doy, seconds = (int(n) for n in split_lines[1].split(':'))
                # Break iteration if time from line in file does not match date reported in filename
                if doy != doyFromFile:
                    logger.warning(
                        'time %s from line in conflict with time %s from file '
                        '%s, will continue reading next tarfile(s)',
                        doy,
                        doyFromFile,
                        j,
                    )
                    continue
                # convert units from mm to m
                d.append(trotot * 0.001)
                Sig.append(trototSD * 0.001)
                dwet.append(trwet * 0.001)
                dhydro.append((trotot - trwet) * 0.001)
                timesList.append(seconds)
            if 'TROP/SOLUTION' in line:
                flag = True
        del f
        # Break iteration if file contains no data.
        if d == []:
            logger.warning(
                'file %s for station %s is empty, will continue reading next tarfile(s)',
                j, j.split('.')[0]
            )
            continue

        # check for missing times
        true_times = list(range(0, 86400, 300))
        if len(timesList) != len(true_times):
            missing = [t not in timesList for t in true_times]
            mask = np.array(missing)
            delay, sig, wet_delay, hydro_delay = [np.full((288,), np.nan)] * 4
            delay[~mask] = d
            sig[~mask] = Sig
            wet_delay[~mask] = dwet
            hydro_delay[~mask] = dhydro
            times = true_times.copy()
        else:
            delay = np.array(d)
            times = np.array(timesList)
            sig = np.array(Sig)
            wet_delay = np.array(dwet)
            hydro_delay = np.array(dhydro)

        # if time not specified, pass all times
        if returnTime is None:
            filtoutput = {
                'ID': [site] * len(wet_delay),
                'Date': [time] * len(wet_delay),
                'ZTD': delay,
                'wet_delay': wet_delay,
                'hydrostatic_delay': hydro_delay,
                'times': times,
                'sigZTD': sig,
            }
            filtoutput = [{key: value[k] for key, value in filtoutput.items()} for k in range(len(filtoutput['ID']))]
        else:
            index = np.argmin(np.abs(np.array(timesList) - returnTime))
            filtoutput = [
                {
                    'ID': site,
                    'Date': time,
                    'ZTD': delay[index],
                    'wet_delay': wet_delay[index],
                    'hydrostatic_delay': hydro_delay[index],
                    'times': times[index],
                    'sigZTD': sig[index],
                }
            ]
        # setup pandas array and write output to CSV, making sure to update existing CSV.
        filtoutput = pd.DataFrame(filtoutput)
        if Path.exists(filename):
            filtoutput.to_csv(filename, index=False, mode='a', header=False)
        else:
            filtoutput.to_csv(filename, index=False)

    # record all used tar files
    allstationTarfiles.extend([Path(stationFile) / k for k in stationTarlist])
    allstationTarfiles.sort()
    del ziprepo


def get_station_data(inFile, dateList, gps_repo=None, numCPUs=8, outDir=None, returnTime=None) -> None:
    """Pull tropospheric delay data for a given station name."""
    if outDir is None:
        outDir = os.getcwd()

    pathbase = Path(outDir) / 'GPS_delays'
    if not Path.exists(pathbase):
        Path.mkdir(pathbase)

    returnTime = seconds_of_day(returnTime)
    # print warning if not divisible by 3 seconds
    if returnTime % 3 != 0:
        index = np.argmin(np.abs(np.array(list(range(0, 86400, 300))) - returnTime))
        updatedreturnTime = str(dt.timedelta(seconds=list(range(0, 86400, 300))[index]))
        logger.warning(
            'input time %s not divisible by 3 seconds, so next closest time %s will be chosen',
            returnTime,
            updatedreturnTime,
        )
        returnTime = updatedreturnTime

    # get list of station zip files
    inFile_df = pd.read_csv(inFile)
    stationFiles = inFile_df['path'].to_list()
    del inFile_df

    if len(stationFiles) > 0:
        outputfiles = []
        args = []
        # parse delays from UNR
        if gps_repo == 'UNR':
            for sf in stationFiles:
                StationID = Path.name(sf).split('.')[0]
                name = Path(pathbase) / StationID + '_ztd.csv'
                args.append((sf, name, dateList, returnTime))
                outputfiles.append(name)
            # Parallelize remote querying of zenith delays
            with mp.Pool(numCPUs) as multipool:
                multipool.starmap(get_delays_UNR, args)

    # confirm file exists (i.e. valid delays exists for specified time/region).
    outputfiles = [i for i in outputfiles if Path.exists(i)]
    # Consolidate all CSV files into one object
    if len(outputfiles) == 0:
        raise RuntimeError('No valid delays found for specified time/region.')
    name = Path(outDir) / f'{gps_repo}combinedGPS_ztd.csv'
    statsFile = pd.concat([pd.read_csv(i) for i in outputfiles])
    # drop all duplicate lines
    statsFile.drop_duplicates(inplace=True)
    # Convert the above object into a csv file and export
    statsFile.to_csv(name, index=False, encoding='utf-8')
    del statsFile

    # Add lat/lon/height info
    origstatsFile = pd.read_csv(inFile)
    statsFile = pd.read_csv(name)
    statsFile = pd.merge(
        left=statsFile, right=origstatsFile[['ID', 'Lat', 'Lon', 'Hgt_m']], how='left', left_on='ID', right_on='ID'
    )
    # drop all lines with nans and sort by station ID and year
    statsFile.dropna(how='any', inplace=True)
    # drop all duplicate lines
    statsFile.drop_duplicates(inplace=True)
    statsFile.sort_values(['ID', 'Date'])
    statsFile.to_csv(name, index=False)
    del origstatsFile, statsFile


def get_date(stationFile: Union[str, Path]) -> tuple[dt.datetime, int, int]:
    """Extract the date from a station delay file."""
    # find the date info
    year = int(stationFile[1])
    doy = int(stationFile[2])
    date = dt.datetime(year, 1, 1) + dt.timedelta(doy - 1)
    return date, year, doy


def seconds_of_day(returnTime: Union[dt.datetime, str]) -> int:
    """Convert HH:MM:SS format time-tag to seconds of day."""
    if isinstance(returnTime, dt.time):
        h, m, s = returnTime.hour, returnTime.minute, returnTime.second
    else:
        h, m, s = map(int, returnTime.split(':'))
    return h * 3600 + m * 60 + s
