#!/usr/bin/env python3
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
import logging
import multiprocessing
import os
import zipfile

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)


def get_delays(stationFile, filename, returnTime=None):
    '''
    Parses and returns a dictionary containing either (1) all
    the GPS delays, if returnTime is None, or (2) only the delay
    at the closest times to to returnTime.
    Inputs:
         stationFile - a .gz station delay file
         returnTime  - specified time of GPS delay
    Outputs:
         a dict and CSV file containing the times and delay information
         (delay in mm, delay uncertainty, delay gradients)
    *NOTE: Due to a formatting error in the tropo SINEX files, the two tropospheric gradient columns
    (TGNTOT and TGETOT) are interchanged, as are the formal error columns (_SIG).
    (source=http://geodesy.unr.edu/gps_timeseries/README_trop2.txt)
    '''
    # Refer to the following sites to interpret stationFile variable names:
    # ftp://igs.org/pub/data/format/sinex_tropo.txt
    # http://geodesy.unr.edu/gps_timeseries/README_trop2.txt

    # sort through station zip files
    allstationTarfiles = []
    # if URL
    if stationFile.startswith('http'):
        r = requests.get(stationFile)
        ziprepo = zipfile.ZipFile(io.BytesIO(r.content))
    # if downloaded file
    else:
        ziprepo = zipfile.ZipFile(stationFile)
    # iterate through tarfiles
    stationTarlist = ziprepo.namelist()
    stationTarlist.sort()
    for j in stationTarlist:
        f = gzip.open(ziprepo.open(j), 'rb')
        # get the date of the file
        time, yearFromFile, doyFromFile = get_date(os.path.basename(j).split('.'))
        # initialize variables
        d, ngrad, egrad, timesList, Sig = [], [], [], [], []
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
                    trotot, trototSD, trwet, tgetot, tgetotSD, tgntot, tgntotSD, wvapor, wvaporSD, mtemp = \
                        [float(t) for t in split_lines[2:]]
                except:
                    continue
                site = split_lines[0]
                year, doy, seconds = [int(n)
                                      for n in split_lines[1].split(':')]
                # Break iteration if time from line in file does not match date reported in filename
                if doy != doyFromFile:
                    log.warning(
                        'time %s from line in conflict with time %s from file '
                        '%s, will continue reading next tarfile(s)',
                        doy, doyFromFile, j
                    )
                    continue
                d.append(trotot)
                ngrad.append(tgntot)
                egrad.append(tgetot)
                timesList.append(seconds)
                Sig.append(trototSD)
            if 'TROP/SOLUTION' in line:
                flag = True
        del f
        # Break iteration if file contains no data.
        if d == []:
            log.warning(
                'file %s for station %s is empty, will continue reading next '
                'tarfile(s)', j, j.split('.')[0]
            )
            continue

        # check for missing times
        true_times = list(range(0, 86400, 300))
        if len(timesList) != len(true_times):
            missing = [
                True if t not in timesList else False for t in true_times]
            mask = np.array(missing)
            delay, sig, east_grad, north_grad = [np.full((288,), np.nan)] * 4
            delay[~mask] = d
            sig[~mask] = Sig
            east_grad[~mask] = egrad
            north_grad[~mask] = ngrad
            times = true_times.copy()
        else:
            delay = np.array(d)
            times = np.array(timesList)
            sig = np.array(Sig)
            east_grad = np.array(egrad)
            north_grad = np.array(ngrad)

        # if time not specified, pass all times
        if returnTime == None:
            filtoutput = {'ID': [site] * len(north_grad), 'Date': [time] * len(north_grad), 'ZTD': delay, 'north_grad': north_grad,
                          'east_grad': east_grad, 'times': times, 'sigZTD': sig}
            filtoutput = [{key: value[k] for key, value in filtoutput.items()}
                          for k in range(len(filtoutput['ID']))]
        else:
            index = np.argmin(np.abs(np.array(timesList) - returnTime))
            filtoutput = [{'ID': site, 'Date': time, 'ZTD': delay[index], 'north_grad': north_grad[index],
                           'east_grad': east_grad[index], 'times': times[index], 'sigZTD': sig[index]}]
        # setup pandas array and write output to CSV, making sure to update existing CSV.
        filtoutput = pd.DataFrame(filtoutput)
        if not os.path.exists(filename):
            filtoutput.to_csv(filename, index=False)
        else:
            filtoutput.to_csv(filename, index=False, mode='a', header=False)

    # record all used tar files
    allstationTarfiles.extend([os.path.join(stationFile, k)
                               for k in stationTarlist])
    allstationTarfiles.sort()
    del ziprepo

    return


def get_station_data(inFile, numCPUs=8, outDir=None, returnTime=None):
    '''
    Pull tropospheric delay data for a given station name
    '''
    if outDir is None:
        outDir = os.getcwd()

    pathbase = os.path.join(outDir, 'GPS_delays')
    if not os.path.exists(pathbase):
        os.mkdir(pathbase)

    returnTime = seconds_of_day(returnTime)
    # print warning if not divisible by 3 seconds
    if returnTime % 3 != 0:
        index = np.argmin(
            np.abs(np.array(list(range(0, 86400, 300))) - returnTime))
        updatedreturnTime = str(dt.timedelta(
            seconds=list(range(0, 86400, 300))[index]))
        log.warning(
            'input time %s not divisble by 3 seconds, so next closest time %s '
            'will be chosen', returnTime, updatedreturnTime
        )
        returnTime = updatedreturnTime

    # get list of station zip files
    inFile_df = pd.read_csv(inFile)
    stationFiles = inFile_df['path'].to_list()
    del inFile_df

    if len(stationFiles) > 0:
        outputfiles = []
        args = []
        for sf in stationFiles:
            StationID = os.path.basename(sf).split('.')[0]
            name = os.path.join(pathbase, StationID + '_ztd.csv')
            args.append((sf, name, returnTime))
            outputfiles.append(name)
        # Parallelize remote querying of zenith delays
        with multiprocessing.Pool(numCPUs) as multipool:
            multipool.starmap(get_delays, args)

    # Consolidate all CSV files into one object
    name = os.path.join(outDir, 'CombinedGPS_ztd.csv')
    statsFile = pd.concat([pd.read_csv(i) for i in outputfiles])
    # drop all duplicate lines
    statsFile.drop_duplicates(inplace=True)
    # Convert the above object into a csv file and export
    statsFile.to_csv(name, index=False, encoding="utf-8")
    del statsFile

    # Add lat/lon info
    origstatsFile = pd.read_csv(inFile)
    statsFile = pd.read_csv(name)
    statsFile = pd.merge(left=statsFile, right=origstatsFile[['ID', 'Lat', 'Lon']], how='left', left_on='ID', right_on='ID')
    # drop all lines with nans and sort by station ID and year
    statsFile.dropna(how='any', inplace=True)
    statsFile.sort_values(['ID', 'Date'])
    statsFile.to_csv(name, index=False)
    del origstatsFile, statsFile


def get_date(stationFile):
    '''
    extract the date from a station delay file
    '''

    # find the date info
    year = int(stationFile[1])
    doy = int(stationFile[2])
    date = dt.datetime(year, 1, 1) + dt.timedelta(doy - 1)

    return date, year, doy


def seconds_of_day(returnTime):
    '''
    Convert HH:MM:SS format time-tag to seconds of day.
    '''
    h, m, s = map(int, returnTime.split(":"))

    return h * 3600 + m * 60 + s
