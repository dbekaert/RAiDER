import argparse
import datetime
import glob
import os

import pandas as pd


def combineDelayFiles(outName, loc = os.getcwd(), ext='.csv'):
    files = glob.glob(loc + '*' + ext)
    addDateTimeToFiles(files)
    concatDelayFiles(files, colList = ['Datetime', 'ID'], outName = outName)


def addDateTimeToFiles(fileList, force=False):
    ''' Run through a list of files and add the datetime of each file as a column '''
    for f in fileList:
        data = pd.read_csv(f)

        if 'Datetime' in data.columns and not force:
            print('Files already have been processed, pass "force = True" if you want to continue')
            return
        dt = getDateTime(f)
        data['Datetime'] = dt
        data.to_csv(f, index=False)
        del data


def getDateTime(filename):
    ''' Parse a datetime from a RAiDER delay filename '''
    parts = filename.split('_')
    dt = parts[2]
    return datetime.datetime.strptime(dt, '%Y%m%dT%H%M%S')


def concatDelayFiles(fileList, colList = ['Datetime', 'ID'], return_df = False, outName = None):
    ''' 
    Read a list of .csv files containing the same columns and append them 
    together, sorting by specified columns 
    '''
    dfList = []
    for f in fileList:
        dfList.append(pd.read_csv(f))
    df_c = pd.concat(dfList)
    df_c.sort_values(by=colList, inplace = True)
    if return_df or outName is None:
        return df_c
    else:
        df_c.to_csv(outName, index=False)


def mergeDelayFiles(raiderFile, ztdFile, outName = None):
    '''
    Merge a combined RAiDER delays file with a GPS ZTD delay file
    '''
    dfr = pd.read_csv(raiderFile)
    dfz = readZTDFile(ztdFile)
    dfc = dfr.merge(dfz[['ID', 'Datetime', 'ZTD']], how='inner', left_on=['Datetime', 'ID'], right_on=['Datetime', 'ID'], sort=True)
    if outName is None:
        return dfc
    else:
        dfc.to_csv(outName, index=False)


def readZTDFile(filename):
    '''
    Read and parse a GPS zenith delay file
    '''
    data = pd.read_csv(filename, parse_dates=['Date'])
    times = data['times'].apply(lambda x: datetime.timedelta(seconds=x))
    data['Datetime'] = data['Date'] + times
    return data

    
def create_parser():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent("""\
            Combine delay files from a weather model and GPS Zenith delays
            Usage examples:
            raiderCombine.py --gnss UNRCombined_gnss.csv --raiderLoc ERA5/ -o Combined_delays.csv 
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
        '--name',
        '-n' 
        dest='column_name',
        help=dedent("""\
            Name of the column containing RAiDER delays.
            """),
        default='raider'
    )

    p.add_argument(
        '--out',
        '-o' 
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
        mergeDelayFiles(args.raider_file, args.gnss_file, outName = args.out_name)
    else:
        combineDelayFiles(args.raider_file, loc = args.raider_folder)
        mergeDelayFiles(args.raider_file, args.gnss_file, outName = args.out_name)

