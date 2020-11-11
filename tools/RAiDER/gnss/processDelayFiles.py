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

    
