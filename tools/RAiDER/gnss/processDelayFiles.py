from textwrap import dedent
import numpy as np
import argparse
import datetime
import glob
import os
import re
import math

from tqdm import tqdm

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def combineDelayFiles(outName, loc=os.getcwd(), source='model', ext='.csv', ref=None, col_name='ZTD'):
    files = glob.glob(os.path.join(loc, '*' + ext))

    if source == 'model':
        print('Ensuring that "Datetime" column exists in files')
        addDateTimeToFiles(files)

    # If single file, just copy source
    if len(files) == 1:
        if source == 'model':
            import shutil
            shutil.copy(files[0], outName)
        else:
            files = readZTDFile(files[0], col_name=col_name)
            # drop all lines with nans
            files.dropna(how='any', inplace=True)
            # drop all duplicate lines
            files.drop_duplicates(inplace=True)
            files.to_csv(outName, index=False)
        return

    print('Combining {} delay files'.format(source))
    try:
        concatDelayFiles(
            files,
            sort_list=['ID', 'Datetime'],
            outName=outName,
            source=source
        )
    except:
        concatDelayFiles(
            files,
            sort_list=['ID', 'Date'],
            outName=outName,
            source=source,
            ref=ref,
            col_name=col_name
        )


def addDateTimeToFiles(fileList, force=False, verbose=False):
    ''' Run through a list of files and add the datetime of each file as a column '''

    print('Adding Datetime to delay files')

    for f in tqdm(fileList):
        data = pd.read_csv(f)

        if 'Datetime' in data.columns and not force:
            if verbose:
                print(
                    'File {} already has a "Datetime" column, pass'
                    '"force = True" if you want to override and '
                    're-process'.format(f)
                )
        else:
            try:
                dt = getDateTime(f)
                data['Datetime'] = dt
                # drop all lines with nans
                data.dropna(how='any', inplace=True)
                # drop all duplicate lines
                data.drop_duplicates(inplace=True)
                data.to_csv(f, index=False)
            except (AttributeError, ValueError):
                print(
                    'File {} does not contain datetime info, skipping'
                    .format(f)
                )
        del data


def getDateTime(filename):
    ''' Parse a datetime from a RAiDER delay filename '''
    filename = os.path.basename(filename)
    dtr = re.compile(r'\d{8}T\d{6}')
    dt = dtr.search(filename)
    return datetime.datetime.strptime(
        dt.group(),
        '%Y%m%dT%H%M%S'
    )


def update_time(row, localTime_hrs):
    '''Update with local origin time'''
    localTime_estimate = row['Datetime'].replace(hour=localTime_hrs, \
                                                 minute=0, second=0)
    # determine if you need to shift days
    time_shift = datetime.timedelta(days=0)
    # round to nearest hour
    days_diff = (row['Datetime'] -
                 datetime.timedelta(seconds=math.floor( \
                 row['Localtime'] )*3600)).day - \
                 localTime_estimate.day
    # if lon <0, check if you need to add day
    if row['Lon'] < 0:
        # add day
        if days_diff != 0:
            time_shift = datetime.timedelta(days=1)
    # if lon >0, check if you need to subtract day
    if row['Lon'] > 0:
        # subtract day
        if days_diff != 0:
            time_shift = -datetime.timedelta(days=1)
    return localTime_estimate + datetime.timedelta(seconds=row['Localtime'] \
                                                   * 3600) + time_shift


def pass_common_obs(reference, target, localtime=None):
    '''Pass only observations in target spatiotemporally common to reference'''
    if localtime:
        return target[target['Datetime'].dt.date.isin(reference['Datetime']
                      .dt.date) &
                      target['ID'].isin(reference['ID']) &
                      target[localtime].isin(reference[localtime])]
    else:
        return target[target['Datetime'].dt.date.isin(reference['Datetime']
                      .dt.date) &
                      target['ID'].isin(reference['ID'])]


def concatDelayFiles(
    fileList,
    sort_list=['ID', 'Datetime'],
    return_df=False,
    outName=None,
    source='model',
    ref=None,
    col_name='ZTD'
):
    ''' 
    Read a list of .csv files containing the same columns and append them 
    together, sorting by specified columns 
    '''
    dfList = []

    print('Concatenating delay files')

    for f in tqdm(fileList):
        if source == 'model':
            dfList.append(pd.read_csv(f, parse_dates=['Datetime']))
        else:
            dfList.append(readZTDFile(f, col_name=col_name))
    # drop lines not found in reference file
    if ref:
        dfr = pd.read_csv(ref, parse_dates=['Datetime'])
        for i in enumerate(dfList):
            dfList[i[0]] = pass_common_obs(dfr, i[1])
        del dfr

    df_c = pd.concat(
        dfList,
        ignore_index=True
    ).drop_duplicates().reset_index(drop=True)
    df_c.sort_values(by=sort_list, inplace=True)

    print('Total number of rows in the concatenated file: {}'.format(df_c.shape[0]))
    print('Total number of rows containing NaNs: {}'.format(
        df_c[df_c.isna().any(axis=1)].shape[0]
    )
    )

    if return_df or outName is None:
        return df_c
    else:
        # drop all lines with nans
        df_c.dropna(how='any', inplace=True)
        # drop all duplicate lines
        df_c.drop_duplicates(inplace=True)
        df_c.to_csv(outName, index=False)


def local_time_filter(raiderFile, ztdFile, dfr, dfz, localTime):
    '''
    Convert to local-time reference frame WRT 0 longitude
    '''
    localTime_hrs = int(localTime.split(' ')[0])
    localTime_hrthreshold = int(localTime.split(' ')[1])
    # with rotation rate and distance to 0 lon, get localtime shift WRT 00 UTC at 0 lon
    # *rotation rate at given point = (360deg/23.9333333333hr) = 15.041782729825965 deg/hr
    dfr['Localtime'] = (dfr['Lon'] / 15.041782729825965)
    dfz['Localtime'] = (dfz['Lon'] / 15.041782729825965)

    # estimate local-times
    dfr['Localtime'] = dfr.apply(lambda r: update_time(r, localTime_hrs),
                                 axis=1)
    dfz['Localtime'] = dfz.apply(lambda r: update_time(r, localTime_hrs),
                                 axis=1)

    # filter out data outside of --localtime hour threshold
    dfr['Localtime_u'] = dfr['Localtime'] + \
                         datetime.timedelta(hours=localTime_hrthreshold)
    dfr['Localtime_l'] = dfr['Localtime'] - \
                         datetime.timedelta(hours=localTime_hrthreshold)
    OG_total = dfr.shape[0]
    dfr = dfr[(dfr['Datetime'] >= dfr['Localtime_l']) &
              (dfr['Datetime'] <= dfr['Localtime_u'])]

    # only keep observation closest to Localtime
    print('Total number of datapoints dropped in {} for not being within '
          '{} hrs of specified local-time {}: {} out of {}'.format(
          raiderFile, localTime.split(' ')[1], localTime.split(' ')[0],
          dfr.shape[0], OG_total))
    dfz['Localtime_u'] = dfz['Localtime'] + \
                         datetime.timedelta(hours=localTime_hrthreshold)
    dfz['Localtime_l'] = dfz['Localtime'] - \
                         datetime.timedelta(hours=localTime_hrthreshold)
    OG_total = dfz.shape[0]
    dfz = dfz[(dfz['Datetime'] >= dfz['Localtime_l']) &
              (dfz['Datetime'] <= dfz['Localtime_u'])]
    # only keep observation closest to Localtime
    print('Total number of datapoints dropped in {} for not being within '
          '{} hrs of specified local-time {}: {} out of {}'.format(
          ztdFile, localTime.split(' ')[1], localTime.split(' ')[0],
          dfz.shape[0], OG_total))

    # drop all lines with nans
    dfr.dropna(how='any', inplace=True)
    dfz.dropna(how='any', inplace=True)
    # drop all duplicate lines
    dfr.drop_duplicates(inplace=True)
    dfz.drop_duplicates(inplace=True)
    # drop and rename columns
    dfr.drop(columns=['Localtime_l', 'Localtime_u'], inplace=True)
    dfz.drop(columns=['Localtime_l', 'Localtime_u'], inplace=True)

    return dfr, dfz


def mergeDelayFiles(
    raiderFile,
    ztdFile,
    col_name='ZTD',
    raider_delay='totalDelay',
    outName=None,
    localTime=None
):
    '''
    Merge a combined RAiDER delays file with a GPS ZTD delay file
    '''
    print('Merging delay files {} and {}'.format(raiderFile, ztdFile))
    dfr = pd.read_csv(raiderFile, parse_dates=['Datetime'])
    # drop extra columns
    expected_data_columns = ['ID','Lat','Lon','Hgt_m','Datetime','wetDelay',
                             'hydroDelay',raider_delay]
    dfr = dfr.drop(columns=[col for col in dfr if col not in \
                            expected_data_columns])
    dfz = pd.read_csv(ztdFile, parse_dates=['Datetime'])
    # drop extra columns
    expected_data_columns = ['ID','Date','wet_delay','hydrostatic_delay',
                             'times','sigZTD','Lat','Lon','Hgt_m','Datetime',
                             col_name]
    dfz = dfz.drop(columns=[col for col in dfz if col not in \
                            expected_data_columns])
    # only pass common locations and times
    dfz = pass_common_obs(dfr, dfz)
    dfr = pass_common_obs(dfz, dfr)

    # If specified, convert to local-time reference frame WRT 0 longitude
    common_keys = ['Datetime', 'ID']
    if localTime is not None:
        dfr, dfz = local_time_filter(raiderFile, ztdFile, dfr, dfz, localTime)
        common_keys.append('Localtime')
        # only pass common locations and times
        dfz = pass_common_obs(dfr, dfz, localtime='Localtime')
        dfr = pass_common_obs(dfz, dfr, localtime='Localtime')

    # drop all lines with nans
    dfr.dropna(how='any', inplace=True)
    dfz.dropna(how='any', inplace=True)
    # drop all duplicate lines
    dfr.drop_duplicates(inplace=True)
    dfz.drop_duplicates(inplace=True)

    print('Beginning merge')

    dfc = dfr.merge(
        dfz[common_keys + ['ZTD', 'sigZTD']],
        how='left',
        left_on=common_keys,
        right_on=common_keys,
        sort=True
    )

    # only keep observation closest to Localtime
    if 'Localtime' in dfc.keys():
        dfc['Localtimediff'] = abs((dfc['Datetime'] - \
                               dfc['Localtime']).dt.total_seconds() / 3600)
        dfc = dfc.loc[dfc.groupby(['ID','Localtime']).Localtimediff.idxmin() \
                      ].reset_index(drop=True)
        dfc.drop(columns=['Localtimediff'], inplace=True)

    # estimate residual
    dfc['ZTD_minus_RAiDER'] = dfc['ZTD'] - dfc[raider_delay]

    print('Total number of rows in the concatenated file: '
          '{}'.format(dfc.shape[0]))
    print('Total number of rows containing NaNs: {}'.format(
        dfc[dfc.isna().any(axis=1)].shape[0]
    )
    )
    print('Merge finished')

    if outName is None:
        return dfc
    else:
        # drop all lines with nans
        dfc.dropna(how='any', inplace=True)
        # drop all duplicate lines
        dfc.drop_duplicates(inplace=True)
        dfc.to_csv(outName, index=False)


def readZTDFile(filename, col_name='ZTD'):
    '''
    Read and parse a GPS zenith delay file
    '''
    try:
        data = pd.read_csv(filename, parse_dates=['Date'])
        times = data['times'].apply(lambda x: datetime.timedelta(seconds=x))
        data['Datetime'] = data['Date'] + times
    except (KeyError, ValueError) as e:
        data = pd.read_csv(filename, parse_dates=['Datetime'])

    data.rename(columns={col_name: 'ZTD'}, inplace=True)
    return data


def create_parser():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent("""\
            Combine delay files from a weather model and GPS Zenith delays
            Usage examples:
            raiderCombine.py --raiderDir './*' --raider 'combined_raider_delays.csv'
            raiderCombine.py  --raiderDir ERA5/ --raider ERA5_combined_delays.csv --raider_column totalDelay --gnssDir GNSS/ --gnss UNRCombined_gnss.csv --column ZTD -o Combined_delays.csv
            raiderCombine.py  --raiderDir ERA5_2019/ --raider ERA5_combined_delays_2019.csv --raider_column totalDelay --gnssDir GNSS_2019/ --gnss UNRCombined_gnss_2019.csv --column ZTD -o Combined_delays_2019_UTTC18.csv --localtime '18:00:00 1'
            """)
    )

    p.add_argument(
        '--raider', dest='raider_file',
        help=dedent("""\
            .csv file containing RAiDER-derived Zenith Delays. 
            Should contain columns "ID" and "Datetime" in addition to the delay column
            If the file does not exist, I will attempt to create it from a directory of 
            delay files. 
            """),
        required=True
    )
    p.add_argument(
        '--raiderDir', '-d', dest='raider_folder',
        help=dedent("""\
            Directory containing RAiDER-derived Zenith Delay files.  
            Files should be named with a Datetime in the name and contain the 
            column "ID" as the delay column names.
            """),
        default=os.getcwd()
    )
    p.add_argument(
        '--gnssDir', '-gd', dest='gnss_folder',
        help=dedent("""\
            Directory containing GNSS-derived Zenith Delay files.  
            Files should contain the column "ID" as the delay column names 
            and times should be denoted by the "Date" key.
            """),
        default=os.getcwd()
    )

    p.add_argument(
        '--gnss', dest='gnss_file',
        help=dedent("""\
            Optional .csv file containing GPS Zenith Delays. Should contain columns "ID", "ZTD", and "Datetime"
            """),
        default=None
    )

    p.add_argument(
        '--raider_column',
        '-r',
        dest='raider_column_name',
        help=dedent("""\
            Name of the column containing RAiDER delays. Only used with the "--gnss" option
            """),
        default='totalDelay'
    )
    p.add_argument(
        '--column',
        '-c',
        dest='column_name',
        help=dedent("""\
            Name of the column containing GPS Zenith delays. Only used with the "--gnss" option

            """),
        default='ZTD'
    )

    p.add_argument(
        '--out',
        '-o',
        dest='out_name',
        help=dedent("""\
            Name to use for the combined delay file. Only used with the "--gnss" option

            """),
        default='Combined_delays.csv'
    )

    p.add_argument(
        '--localtime',
        '-lt',
        dest='local_time',
        help=dedent("""\
            "Optional control to pass only data at local-time (in integer hours) WRT user-defined time at 0 longitude (1st argument),
             and within +/- specified hour threshold (2nd argument).
             By default UTC is passed as is without local-time conversions.
             Input in 'HH H', e.g. '16 1'"

            """),
        default=None
    )

    return p


def parseCMD():
    """
    Parse command-line arguments and pass to tropo_delay
    We'll parse arguments and call delay.py.
    """

    p = create_parser()
    args = p.parse_args()

    if not os.path.exists(args.raider_file):
        combineDelayFiles(args.raider_file, loc=args.raider_folder)

    if not os.path.exists(args.gnss_file):
        combineDelayFiles(args.gnss_file, loc=args.gnss_folder, source='GNSS',
                          ref=args.raider_file, col_name=args.column_name)

    if args.gnss_file is not None:
        mergeDelayFiles(
            args.raider_file,
            args.gnss_file,
            col_name=args.column_name,
            raider_delay=args.raider_column_name,
            outName=args.out_name,
            localTime=args.local_time
        )
