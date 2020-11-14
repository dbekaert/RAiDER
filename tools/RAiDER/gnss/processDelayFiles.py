import argparse
import datetime
import glob
import os
from tqdm import tqdm

import pandas as pd

from textwrap import dedent


def combineDelayFiles(outName, loc = os.getcwd(), ext='.csv'):
    files = glob.glob(os.path.join(loc, '*' + ext))

    print('Ensuring that "Datetime" column exists in files')

    addDateTimeToFiles(files)

    print('Combining weather model delay files')
    concatDelayFiles(files, sort_list = ['Datetime', 'ID'], outName = outName)


def addDateTimeToFiles(fileList, force=False):
    ''' Run through a list of files and add the datetime of each file as a column '''

    print('Adding Datetime to delay files')

    for f in tqdm(fileList):
        data = pd.read_csv(f)

        if 'Datetime' in data.columns and not force:
            print('Files already have a "Datetime" column, pass "force = True" if you want to override and re-process')
            return
        dt = getDateTime(f)
        data['Datetime'] = dt
        data.to_csv(f, index=False)
        del data


def getDateTime(filename):
    ''' Parse a datetime from a RAiDER delay filename '''
    filename = os.path.basename(filename)
    parts = filename.split('_')
    dt = parts[2]
    return datetime.datetime.strptime(dt, '%Y%m%dT%H%M%S')


def concatDelayFiles(fileList, sort_list = ['Datetime', 'ID'], return_df = False, outName = None):
    ''' 
    Read a list of .csv files containing the same columns and append them 
    together, sorting by specified columns 
    '''
    dfList = []
    
    print('Concatenating delay files')

    for f in tqdm(fileList):
        dfList.append(pd.read_csv(f))
    
    df_c = pd.concat(dfList, ignore_index=True).drop_duplicates().reset_index(drop=True)
    df_c.sort_values(by=sort_list, inplace = True)

    if return_df or outName is None:
        return df_c
    else:
        df_c.to_csv(outName, index=False)


def mergeDelayFiles(raiderFile, ztdFile, col_name = 'ZTD', raider_delay = 'totalDelay', outName = None):
    '''
    Merge a combined RAiDER delays file with a GPS ZTD delay file
    '''
    
    print('Merging delay files {} and {}'.format(raiderFile, ztdFile))

    dfr = pd.read_csv(raiderFile, parse_dates=['Datetime'])
    dfz = readZTDFile(ztdFile, col_name = col_name)

    print('Beginning merge')

    dfc = dfr.merge(dfz[['ID', 'Datetime', 'ZTD']], how='inner', left_on=['Datetime', 'ID'], right_on=['Datetime', 'ID'], sort=True)
    dfc['ZTD_minus_RAiDER'] = dfc['ZTD'] - dfc[raider_delay]

    print('Merge finished')

    if outName is None:
        return dfc
    else:
        dfc.to_csv(outName, index=False)


def readZTDFile(filename, col_name = 'ZTD'):
    '''
    Read and parse a GPS zenith delay file
    '''
    try:
        data = pd.read_csv(filename, parse_dates=['Date'])
        times = data['times'].apply(lambda x: datetime.timedelta(seconds=x))
        data['Datetime'] = data['Date'] + times
    except KeyError:
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
            raiderCombine.py --gnss UNRCombined_gnss.csv --raiderLoc ERA5/ --raider ERA5_combined_delays.csv -o Combined_delays.csv --column ERA5
            """)
    )

    p.add_argument(
        '--gnss', dest='gnss_file',
        help=dedent("""\
            .csv file containing GPS Zenith Delays. Should contain columns "ID", "ZTD", and "Datetime"
            """),
        required=True
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
        default = os.getcwd()
    )

    p.add_argument(
        '--raider_column',
        '-r',
        dest='raider_column_name',
        help=dedent("""\
            Name of the column containing RAiDER delays.
            """),
        default='totalDelay'
    )
    p.add_argument(
        '--column',
        '-c',
        dest='column_name',
        help=dedent("""\
            Name of the column containing GPS Zenith delays.
            """),
        default='ZTD'
    )

    p.add_argument(
        '--out',
        '-o', 
        dest='out_name',
        help=dedent("""\
            Name to use for the combined delay file
            """),
        default='Combined_delays.csv'
    )

    return p


def parseCMD():
    """
    Parse command-line arguments and pass to tropo_delay
    We'll parse arguments and call delay.py.
    """

    p = create_parser()
    args = p.parse_args()

    if os.path.exists(args.raider_file):
        mergeDelayFiles(args.raider_file, args.gnss_file, col_name = args.column_name, raider_delay = args.raider_column_name, outName = args.out_name)
    else:
        combineDelayFiles(args.raider_file, loc = args.raider_folder)
        mergeDelayFiles(args.raider_file, args.gnss_file, col_name = args.column_name, raider_delay = args.raider_column_name, outName = args.out_name)

