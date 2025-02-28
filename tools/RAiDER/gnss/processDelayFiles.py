import argparse
import datetime as dt
import math
import re
from pathlib import Path
from textwrap import dedent
from typing import Optional

import pandas as pd
from tqdm import tqdm


pd.options.mode.chained_assignment = None  # default='warn'


def combineDelayFiles(
    out_path: Path,
    loc: Path=Path.cwd(),
    source: str='model',
    ext: str='.csv',
    ref: Optional[Path]=None,
    col_name: str='ZTD'
) -> None:
    file_paths = list(loc.glob('*' + ext))

    if source == 'model':
        print('Ensuring that "Datetime" column exists in files')
        addDateTimeToFiles(file_paths)

    # If single file, just copy source
    if len(file_paths) == 1:
        if source == 'model':
            import shutil
            shutil.copy(file_paths[0], out_path)
        else:
            file_paths = readZTDFile(file_paths[0], col_name=col_name)
            # drop all lines with nans
            file_paths.dropna(how='any', inplace=True)
            # drop all duplicate lines
            file_paths.drop_duplicates(inplace=True)
            file_paths.to_csv(out_path, index=False)
        return

    print(f'Combining {source} delay files')
    try:
        concatDelayFiles(file_paths, sort_list=['ID', 'Datetime'], outName=out_path, source=source)
    except:
        concatDelayFiles(file_paths, sort_list=['ID', 'Date'], outName=out_path, source=source, ref=ref, col_name=col_name)


def addDateTimeToFiles(file_paths: list[Path], force: bool=False, verbose: bool=False) -> None:
    """Run through a list of files and add the datetime of each file as a column."""
    print('Adding Datetime to delay files')

    for path in tqdm(file_paths):
        data = pd.read_csv(path)

        if 'Datetime' in data.columns and not force:
            if verbose:
                print(
                    f'File {path} already has a "Datetime" column, pass'
                    '"force = True" if you want to override and '
                    're-process'
                )
        else:
            try:
                data['Datetime'] = getDateTime(path)
                # drop all lines with nans
                data.dropna(how='any', inplace=True)
                # drop all duplicate lines
                data.drop_duplicates(inplace=True)
                data.to_csv(path, index=False)
            except (AttributeError, ValueError):
                print(f'File {path} does not contain datetime info, skipping')
        del data


def getDateTime(path: Path) -> dt.datetime:
    """Parse a datetime from a RAiDER delay filename."""
    datetime_pattern = re.compile(r'\d{8}T\d{6}')
    match = datetime_pattern.search(path.name)
    return dt.datetime.strptime(match.group(), '%Y%m%dT%H%M%S')


def update_time(row, localTime_hrs):
    """Update with local origin time."""
    localTime_estimate = row['Datetime'].replace(hour=localTime_hrs, minute=0, second=0)
    # determine if you need to shift days
    time_shift = dt.timedelta(days=0)
    # round to nearest hour
    days_diff = (
        row['Datetime'] - dt.timedelta(seconds=math.floor(row['Localtime']) * 3600)
    ).day - localTime_estimate.day
    # if lon <0, check if you need to add day
    if row['Lon'] < 0:
        # add day
        if days_diff != 0:
            time_shift = dt.timedelta(days=1)
    # if lon >0, check if you need to subtract day
    if row['Lon'] > 0:
        # subtract day
        if days_diff != 0:
            time_shift = -dt.timedelta(days=1)
    return localTime_estimate + dt.timedelta(seconds=row['Localtime'] * 3600) + time_shift


def pass_common_obs(reference, target, localtime=None):
    """Pass only observations in target spatiotemporally common to reference."""
    if isinstance(target['Datetime'].iloc[0], str):
        target['Datetime'] = target['Datetime'].apply(
            lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        )
    if localtime:
        return target[
            target['Datetime'].dt.date.isin(reference['Datetime'].dt.date)
            & target['ID'].isin(reference['ID'])
            & target[localtime].isin(reference[localtime])
        ]
    else:
        return target[
            target['Datetime'].dt.date.isin(reference['Datetime'].dt.date) &
            target['ID'].isin(reference['ID'])
        ]


def concatDelayFiles(
    fileList, sort_list=['ID', 'Datetime'], return_df=False, outName=None, source='model', ref=None, col_name='ZTD'
):
    """
    Read a list of .csv files containing the same columns and append them
    together, sorting by specified columns.
    """
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

    df_c = pd.concat(dfList, ignore_index=True).drop_duplicates().reset_index(drop=True)
    df_c.sort_values(by=sort_list, inplace=True)

    print(f'Total number of rows in the concatenated file: {df_c.shape[0]}')
    print(f'Total number of rows containing NaNs: {df_c[df_c.isna().any(axis=1)].shape[0]}')

    if return_df or outName is None:
        return df_c
    else:
        # drop all lines with nans
        df_c.dropna(how='any', inplace=True)
        # drop all duplicate lines
        df_c.drop_duplicates(inplace=True)
        df_c.to_csv(outName, index=False)


def local_time_filter(raiderFile, ztdFile, dfr, dfz, localTime):
    """Convert to local-time reference frame WRT 0 longitude."""
    localTime_hrs = int(localTime.split(' ')[0])
    localTime_hrthreshold = int(localTime.split(' ')[1])
    # with rotation rate and distance to 0 lon, get localtime shift WRT 00 UTC at 0 lon
    # *rotation rate at given point = (360deg/23.9333333333hr) = 15.041782729825965 deg/hr
    dfr['Localtime'] = dfr['Lon'] / 15.041782729825965
    dfz['Localtime'] = dfz['Lon'] / 15.041782729825965

    # estimate local-times
    dfr['Localtime'] = dfr.apply(lambda r: update_time(r, localTime_hrs), axis=1)
    dfz['Localtime'] = dfz.apply(lambda r: update_time(r, localTime_hrs), axis=1)

    # filter out data outside of --localtime hour threshold
    dfr['Localtime_u'] = dfr['Localtime'] + dt.timedelta(hours=localTime_hrthreshold)
    dfr['Localtime_l'] = dfr['Localtime'] - dt.timedelta(hours=localTime_hrthreshold)
    OG_total = dfr.shape[0]
    dfr = dfr[(dfr['Datetime'] >= dfr['Localtime_l']) & (dfr['Datetime'] <= dfr['Localtime_u'])]

    # only keep observation closest to Localtime
    print(
        f'Total number of datapoints dropped in {raiderFile} for not being within {localTime.split(" ")[1]} hrs of '
        f'specified local-time {localTime.split(" ")[0]}: {dfr.shape[0]} out of {OG_total}'
    )
    dfz['Localtime_u'] = dfz['Localtime'] + dt.timedelta(hours=localTime_hrthreshold)
    dfz['Localtime_l'] = dfz['Localtime'] - dt.timedelta(hours=localTime_hrthreshold)
    OG_total = dfz.shape[0]
    dfz = dfz[(dfz['Datetime'] >= dfz['Localtime_l']) & (dfz['Datetime'] <= dfz['Localtime_u'])]
    # only keep observation closest to Localtime
    print(
        f'Total number of datapoints dropped in {ztdFile} for not being within {localTime.split(" ")[1]} hrs of '
        f'specified local-time {localTime.split(" ")[0]}: {dfz.shape[0]} out of {OG_total}'
    )

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


def readZTDFile(filename, col_name='ZTD'):
    """Read and parse a GPS zenith delay file."""
    try:
        data = pd.read_csv(filename, parse_dates=['Date'])
        times = data['times'].apply(lambda x: dt.timedelta(seconds=x))
        data['Datetime'] = data['Date'] + times
    except (KeyError, ValueError):
        data = pd.read_csv(filename, parse_dates=['Datetime'])

    data.rename(columns={col_name: 'ZTD'}, inplace=True)
    return data


def file_choices(p: argparse.ArgumentParser, choices: tuple[str], s: str) -> Path:
    path = Path(s)
    if path.suffix not in choices:
       p.error(f"File must end with one of {choices}")
    return path

def parse_dir(p: argparse.ArgumentParser, s: str) -> Path:
    path = Path(s)
    if not path.is_dir():
        p.error("Path must be a directory")
    return path


def create_parser() -> argparse.ArgumentParser:
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent("""\
            Combine delay files from a weather model and GPS Zenith delays
            Usage examples:
            raiderCombine.py --raiderDir './*' --raider 'combined_raider_delays.csv'
            raiderCombine.py  --raiderDir ERA5/ --raider ERA5_combined_delays.csv --raider_column totalDelay --gnssDir GNSS/ --gnss UNRCombined_gnss.csv --column ZTD -o Combined_delays.csv
            raiderCombine.py  --raiderDir ERA5_2019/ --raider ERA5_combined_delays_2019.csv --raider_column totalDelay --gnssDir GNSS_2019/ --gnss UNRCombined_gnss_2019.csv --column ZTD -o Combined_delays_2019_UTTC18.csv --localtime '18:00:00 1'
            """),
    )

    p.add_argument(
        '--raider',
        dest='raider_file',
        help=dedent("""\
            .csv file containing RAiDER-derived Zenith Delays.
            Should contain columns "ID" and "Datetime" in addition to the delay column
            If the file does not exist, I will attempt to create it from a directory of
            delay files.
            """),
        required=True,
        type=lambda s: file_choices(p, ('csv',), s),
    )
    p.add_argument(
        '--raiderDir',
        '-d',
        dest='raider_folder',
        help=dedent("""\
            Directory containing RAiDER-derived Zenith Delay files.
            Files should be named with a Datetime in the name and contain the
            column "ID" as the delay column names.
            """),
        type=lambda s: parse_dir(p, s),
        default=Path.cwd(),
    )
    p.add_argument(
        '--gnssDir',
        '-gd',
        dest='gnss_folder',
        help=dedent("""\
            Directory containing GNSS-derived Zenith Delay files.
            Files should contain the column "ID" as the delay column names
            and times should be denoted by the "Date" key.
            """),
        type=lambda s: parse_dir(p, s),
        default=Path.cwd(),
    )

    p.add_argument(
        '--gnss',
        dest='gnss_file',
        help=dedent("""\
            Optional .csv file containing GPS Zenith Delays. Should contain columns "ID", "ZTD", and "Datetime"
            """),
        default=None,
        type=lambda s: file_choices(p, ('csv',), s),
    )

    p.add_argument(
        '--raider_column',
        '-r',
        dest='raider_column_name',
        help=dedent("""\
            Name of the column containing RAiDER delays. Only used with the "--gnss" option
            """),
        default='totalDelay',
    )
    p.add_argument(
        '--column',
        '-c',
        dest='column_name',
        help=dedent("""\
            Name of the column containing GPS Zenith delays. Only used with the "--gnss" option

            """),
        default='ZTD',
    )

    p.add_argument(
        '--out',
        '-o',
        dest='out_name',
        help=dedent("""\
            Name to use for the combined delay file. Only used with the "--gnss" option
            """),
        type=Path,
        default=Path('Combined_delays.csv'),
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
        default=None,
    )

    return p


def main(
    raider_file: Path,
    ztd_file: Path,
    col_name: str='ZTD',
    raider_delay: str='totalDelay',
    out_path: Optional[Path]=None,
    local_time=None
):
    """Merge a combined RAiDER delays file with a GPS ZTD delay file."""
    print(f'Merging delay files {raider_file} and {ztd_file}')
    dfr = pd.read_csv(raider_file, parse_dates=['Datetime'])
    # drop extra columns
    expected_data_columns = ['ID', 'Lat', 'Lon', 'Hgt_m', 'Datetime', 'wetDelay', 'hydroDelay', raider_delay]
    dfr = dfr.drop(columns=[col for col in dfr if col not in expected_data_columns])
    dfz = pd.read_csv(ztd_file, parse_dates=['Date'])
    if 'Datetime' not in dfz.keys():
        dfz.rename(columns={'Date': 'Datetime'}, inplace=True)
    # drop extra columns
    expected_data_columns = [
        'ID',
        'Datetime',
        'wet_delay',
        'hydrostatic_delay',
        'times',
        'sigZTD',
        'Lat',
        'Lon',
        'Hgt_m',
        col_name,
    ]
    dfz = dfz.drop(columns=[col for col in dfz if col not in expected_data_columns])
    # only pass common locations and times
    dfz = pass_common_obs(dfr, dfz)
    dfr = pass_common_obs(dfz, dfr)

    # If specified, convert to local-time reference frame WRT 0 longitude
    common_keys = ['Datetime', 'ID']
    if local_time is not None:
        dfr, dfz = local_time_filter(raider_file, ztd_file, dfr, dfz, local_time)
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
        dfz[common_keys + ['ZTD', 'sigZTD']], how='left', left_on=common_keys, right_on=common_keys, sort=True
    )

    # only keep observation closest to Localtime
    if 'Localtime' in dfc.keys():
        dfc['Localtimediff'] = abs((dfc['Datetime'] - dfc['Localtime']).dt.total_seconds() / 3600)
        dfc = dfc.loc[dfc.groupby(['ID', 'Localtime']).Localtimediff.idxmin()].reset_index(drop=True)
        dfc.drop(columns=['Localtimediff'], inplace=True)

    # estimate residual
    dfc['ZTD_minus_RAiDER'] = dfc['ZTD'] - dfc[raider_delay]

    print('Total number of rows in the concatenated file: ' f'{dfc.shape[0]}')
    print(f'Total number of rows containing NaNs: {dfc[dfc.isna().any(axis=1)].shape[0]}')
    print('Merge finished')

    if out_path is None:
        return dfc
    else:
        # drop all lines with nans
        dfc.dropna(how='any', inplace=True)
        # drop all duplicate lines
        dfc.drop_duplicates(inplace=True)
        dfc.to_csv(out_path, index=False)
