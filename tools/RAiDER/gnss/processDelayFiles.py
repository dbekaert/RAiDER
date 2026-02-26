# Standard library
import argparse
import datetime as dt
import glob
import math
import re
import shutil
from itertools import chain
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Union

# Third-party
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local
from RAiDER.cli.parser import add_allow_nan_options, add_verbose
from RAiDER.logger import logger


pd.options.mode.chained_assignment = None  # default='warn'


def combineDelayFiles(
    out_path: Path,
    loc: Union[List[Path], Path] = Path.cwd(),
    source: str='model',
    ext: str='.csv',
    ref: Optional[Path]=None,
    col_name: str='ZTD'
) -> None:

    # Normalize single Path to List
    # e.g. Path('folder') -> [Path('folder')]
    if isinstance(loc, Path):
        loc = [loc]

    # Flatten nested lists if they exist
    # e.g. [[Path('A')], [Path('B')]] -> [Path('A'), Path('B')]
    # This checks if the list is not empty AND the first item is a list
    if loc and isinstance(loc[0], list):
        loc = list(chain.from_iterable(loc))

    # Now 'loc' is guaranteed to be flat: [Path, Path, ...]
    file_paths = [f for folder in loc for f in folder.glob(f"*{ext}")]

    if source == 'model':
        print('Ensuring that "Datetime" column exists in files')
        addDateTimeToFiles(file_paths)

    # If single file, just copy source
    if len(file_paths) == 1:
        if source == 'model':
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
    # modify dataframes without warnings
    dfr = dfr.copy()
    dfz = dfz.copy()

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

    # Ensure Datetime is pandas datetime64[ns] before returning
    for _df in (dfr, dfz):
        _df['Datetime'] = pd.to_datetime(_df['Datetime'], errors='raise')

    return dfr, dfz


def readZTDFile(filename, col_name='ZTD'):
    """Read and parse a GPS zenith delay file."""
    try:
        data = pd.read_csv(filename, parse_dates=['Date'])
        date0 = pd.to_datetime(data['Date'],
            errors='raise',
            format='%Y-%m-%d')

        # If present, convert seconds → pandas Timedelta; otherwise zero
        if 'times' in data.columns:
            sec = pd.to_numeric(data['times'], errors='coerce').fillna(0)
            td = pd.to_timedelta(sec, unit='s')
        else:
            td = pd.to_timedelta(0, unit='s')

        # Combine using numpy/pandas arrays
        # (stays in datetime64[ns], never Python objects)
        dt_vals = date0.values + td.values 

        # Assign back
        data['Datetime'] = pd.to_datetime(dt_vals)
    except (KeyError, ValueError):
        data = pd.read_csv(filename, parse_dates=['Datetime'])

    data.rename(columns={col_name: 'ZTD'}, inplace=True)
    return data


def sampling_delta_stats(df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute global temporal sampling statistics.
    Needed to inform temporal sampling overlap percentage.

    For each station ID:
        * Sort by Datetime.
        * Compute time differences (days) between consecutive observations.

    Then, over all stations combined:
        * Compute the mean time difference in days.
        * Compute the most common time difference (mode) in days.

    Args:
        df: Dataframe with columns "ID" and "Datetime".
            "Datetime" must be datetime-like or parseable as datetime.

    Returns:
        A tuple of:
            mean_delta_days: float
                Mean time difference in days.
            mode_delta_days: float
                Most common time difference in days (global mode).
    """
    # Ensure Datetime is datetime64
    if not np.issubdtype(df["Datetime"].dtype, np.datetime64):
        df = df.copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="raise")

    # Work on a sorted view to get correct diffs per station
    df_sorted = df.sort_values(["ID", "Datetime"])

    # Time differences between consecutive observations per station (in days)
    delta_days = (
        df_sorted.groupby("ID", sort=False)["Datetime"]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
    )

    # Drop NaNs from the first diff in each group
    delta_days = delta_days.dropna()

    if delta_days.empty:
        return np.nan, np.nan

    mean_delta_days = float(delta_days.mean())

    # Global mode of all deltas
    mode_delta_days = float(delta_days.mode().iloc[0])

    return mean_delta_days, mode_delta_days


def variance_analysis(
    group: pd.DataFrame,
    allow_nan_for_negative: bool = True,
    has_localtime: bool = False,
    global_start=None,
    global_end=None,
    n_global_days=None,
) -> pd.Series:
    """
    Compute variance terms and time span for one GNSS station.

    Args:
        group (pd.DataFrame): Subset of rows for a single station ID.
        allow_nan_for_negative (bool): If True, return NaN when
            σ_wm² < 0; otherwise clamp to 0. Default is True.
        has_localtime (bool): If True, parse and output Localtime fields.
        global_start/global_end: Earliest/latest dates across the
            merged dataset, used for percent coverage tracking.
        n_global_days (int): Total number of unique days between
            global_start/global_end (inclusive).

    Returns:
        pd.Series: Summary statistics for this station.
    """

    # Capture wm-gnss residual and sig_ztd
    resid = group["ZTD_minus_RAiDER"] # also mean(R)
    sig = group["sigZTD"]
    n_epochs = len(resid)

    # Parse datetime ranges
    dt = pd.to_datetime(group["Datetime"], errors="coerce")

    # Capture number of observations and unique dates
    unique_dates = (
        dt.dt.normalize()
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    n_unique_days = unique_dates.size
    if n_unique_days > 0:
        station_start = unique_dates.iloc[0]
        station_end = unique_dates.iloc[-1]
    else:
        station_start = pd.NaT
        station_end = pd.NaT
        logger.warning(
            "Flagged station %s with invalid station start and/or end "
            "date(s). Refer to unique dates here: %s.",
            group.name,
            n_unique_days,
        )

    if n_global_days is not None and n_global_days > 0:
        coverage_pct = (n_unique_days / n_global_days) * 100.0
    else:
        coverage_pct = np.nan
        logger.warning(
            "Flagged station %s with no valid sampled days %s.",
            group.name,
            n_global_days,
        )

    # Mean-squared terms
    # σ_res² = D[r] = E((r - E(r))²)
    sigma_res_sq = (
        np.var(resid, ddof=1)
        if n_epochs > 1
        else np.var(resid, ddof=0)
    )
    # σ_GNSS² = E[sigZTD²]
    sigma_gnss_sq = (
        (sig**2).mean()
        if n_epochs > 1
        else np.nan
    )

    # report warnings associated with insufficient sampling
    if not n_epochs > 1:
        logger.warning(
            "Flagged station %s with insufficient valid epochs %d, so σ_res² "
            "& σ_GNSS² set to NaN",
            group.name,
            n_epochs,
        )
    if not len(sig) > 0:
        logger.warning(
            "Flagged station %s with insufficient valid sigZTD observations "
            "%d, so σ_mean² set to NaN",
            group.name,
            len(sig),
        )

    # Mean bias calculation
    mean_bias = resid.mean()

    # Model variance computation
    sigma_model_neg = False
    if np.isfinite(sigma_res_sq) and np.isfinite(sigma_gnss_sq):
        # σ_wm² = σ_res² - σ_gnss²
        diff = sigma_res_sq - sigma_gnss_sq
        negative_diff = diff < 0

        if negative_diff:
            sigma_model_neg = True
            logger.warning(
                "Flagged station %s with negative sigma values, "
                "with mean bias %s, "
                "mean σ_wm² %s, with %d unique days sampled "
                "which translates to %.2f%% daily overlap with the "
                "input dataset timespan.",
                group.name,
                mean_bias,
                diff,
                n_unique_days,
                coverage_pct,
            )
        sigma_model_sq = (
            np.nan
            if negative_diff and allow_nan_for_negative
            else max(diff, 0.0)
        )
    else:
        sigma_model_sq = np.nan

    # Uncertainty in mean bias (error propagation)
    if np.isfinite(sigma_model_sq) and len(sig) > 0:
        sig_squared = sig**2
        # Each measurement has uncertainty: σ_i² = σ_GNSS_i² + σ_model²
        # For mean: σ_mean² = Σ(σ_i²) / n²
        variance_sum = sig_squared.sum() + len(sig) * sigma_model_sq
        sigma_mean_bias = np.sqrt(variance_sum) / len(sig)
    else:
        sigma_mean_bias = np.nan

    def first_non_null(series: pd.Series, label: str):
        vals = series.dropna()
        if not vals.empty:
            return vals.iloc[0]
        else:
            logger.warning(
                "Flagged station %s with no valid %s data.",
                group.name,
                label,
            )
            return np.nan

    data_series = {
        "ID": group.name,
        "Lat": first_non_null(group["Lat"], "Lat"),
        "Lon": first_non_null(group["Lon"], "Lon"),
        "Hgt_m": first_non_null(group["Hgt_m"], "Hgt_m"),
        "Datetime": station_start,
        "Enddate_Datetime": station_end,
        "sigZTD": group["sigZTD"].median(),
        "mean_bias": mean_bias,
        "sigma_mean_bias": sigma_mean_bias,
        "sigma_res": np.sqrt(sigma_res_sq)
        if np.isfinite(sigma_res_sq) else np.nan,
        "sigma_gnss": np.sqrt(sigma_gnss_sq)
        if np.isfinite(sigma_gnss_sq) else np.nan,
        "sigma_model": np.sqrt(sigma_model_sq)
        if np.isfinite(sigma_model_sq) else np.nan,
        "n_epochs": n_epochs,
        "n_unique_days": n_unique_days,
        "pct_days_global": coverage_pct,
        "sigma_model_neg": sigma_model_neg,
    }

    # Only parse Localtime if present
    if has_localtime:
        lt = pd.to_datetime(
            group["Localtime"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )
        data_series.update(
            {
                "Localtime": lt.min(),
                "Enddate_Localtime": lt.max(),
            }
        )

    return pd.Series(data_series)


def file_choices(p: argparse.ArgumentParser, choices: tuple[str], s: str) -> Path:
    path = Path(s)
    if path.suffix not in choices:
       p.error(f"File must end with one of {choices}")
    return path

def parse_dir(pattern: str) -> list[Path]:
    """
    Expand a single directory or wildcard pattern into
    a list of valid Path objects.
    """
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise ValueError(f"No directories found matching pattern: {pattern}")
    return paths


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
        type=lambda s: file_choices(p, ('csv','.csv'), s),
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
        type=parse_dir,
        default=[Path.cwd()],
        nargs='+' # Forces input into a list [Path, Path...]
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
        type=parse_dir,
        default=[Path.cwd()],
        nargs='+' # Forces input into a list [Path, Path...]
    )

    p.add_argument(
        '--gnss',
        dest='gnss_file',
        help=dedent("""\
            Optional .csv file containing GPS Zenith Delays. Should contain columns "ID", "ZTD", and "Datetime"
            """),
        default=None,
        type=lambda s: file_choices(p, ('csv','.csv'), s),
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

    p.add_argument(
        '-oe',
        '--obs_errlimit',
        dest='obs_errlimit',
        help=dedent(
            """\
            Observation error threshold for discarding observations
            with large uncertainties.
            """
        ),
        type=float,
        default=float('inf'),
    )

    p.add_argument(
        '--min-pct-days',
        dest='min_pct_days',
        help=dedent(
            """\
            Minimum pct_days_global threshold required to keep a station in
            the variance CSV. Default 0 means retain all stations.
            """
        ),
        type=float,
        default=0.0,
    )

    p.add_argument(
        '--timeinterval',
        '-ti',
        dest='timeinterval',
        type=str,
        help=dedent("""\
            Subset in time by specifying earliest YYYY-MM-DD date
            followed by latest date YYYY-MM-DD.
            -- Example : '2016-01-01 2019-01-01'."""),
        default=None,
    )

    # add other args to parser
    add_allow_nan_options(p)
    add_verbose(p)

    return p


def main(
    raider_file: Path,
    ztd_file: Path,
    col_name: str='ZTD',
    raider_delay: str='totalDelay',
    out_path: Optional[Path]=None,
    local_time: str=None,
    obs_errlimit: float=float('inf'),
    allow_nan_for_negative: bool=True,
    min_pct_days: float=0.0,
    timeinterval: str=None,
):
    """Merge a combined RAiDER delays file with a GPS ZTD delay file."""
    print(f'Merging delay files {raider_file} and {ztd_file}')

    # load files
    dfz = pd.read_csv(ztd_file, parse_dates=['Datetime'])
    dfr = pd.read_csv(raider_file, parse_dates=['Datetime'])

    # time-interval filter
    # need to add a day buffer to account for time changes
    if timeinterval:
        # Parse the time interval string
        start_str, end_str = timeinterval.split()

        # Convert to datetime objects and apply the 1-day buffer
        # Subtract 1 day from start, Add 1 day to end
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
        start_date_buffer = start_date - pd.Timedelta(days=1)
        end_date_buffer = end_date + pd.Timedelta(days=1)

        # apply time filter
        dfz = dfz[
            (dfz['Datetime'] >= start_date_buffer) & 
            (dfz['Datetime'] <= end_date_buffer)
        ].reset_index(drop=True)
        dfr = dfr[
            (dfr['Datetime'] >= start_date_buffer) & 
            (dfr['Datetime'] <= end_date_buffer)
        ].reset_index(drop=True)

    # drop extra columns from tropo delay file
    expected_data_columns = ['ID', 'Lat', 'Lon', 'Hgt_m', 'Datetime', 'wetDelay', 'hydroDelay', raider_delay]
    dfr = dfr.drop(columns=[col for col in dfr if col not in expected_data_columns])

    # Create dictionaries mapping ID → Lat and ID → Lon from GNSS file
    lat_map = dict(zip(dfz["ID"], dfz["Lat"]))
    lon_map = dict(zip(dfz["ID"], dfz["Lon"]))

    # Apply to tropo delay file to avoid discrepancies in lat/lon
    # since this will lead to dropped matches downstream
    # as localtime estimation would be inconsistent
    dfr = dfr.copy()
    dfr["Lat"] = dfr["ID"].map(lat_map)
    dfr["Lon"] = dfr["ID"].map(lon_map)

    # Round raider datetime to the nearest 5 min to match GNSS data
    dfr['Datetime'] = dfr['Datetime'].apply(
        lambda x: x - dt.timedelta(minutes=x.minute % 5, seconds=x.second, microseconds=x.microsecond)
    )

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

    # use time-interval again to filter based on 'Localtime'
    # to remove straggling observations outside of specified span
    if timeinterval:
        dfz = dfz[
            (dfz['Localtime'] >= start_date) & 
            (dfz['Localtime'] <= end_date)
        ].reset_index(drop=True)
        dfr = dfr[
            (dfr['Localtime'] >= start_date) & 
            (dfr['Localtime'] <= end_date)
        ].reset_index(drop=True)

    # drop all lines with nans
    dfr.dropna(how='any', inplace=True)
    dfz.dropna(how='any', inplace=True)
    # drop all duplicate lines
    dfr.drop_duplicates(inplace=True)
    dfz.drop_duplicates(inplace=True)

    # merge the two dataframes
    print('Beginning merge')
    dfc = dfr.merge(
        dfz[common_keys + ['ZTD', 'sigZTD']], how='left', left_on=common_keys, right_on=common_keys, sort=True
    )

    # only keep observation closest to Localtime
    has_localtime = "Localtime" in dfc.columns
    if has_localtime:
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
        # force consistent datetime format
        dfc['Datetime'] = pd.to_datetime(dfc['Datetime'], errors='raise')
        dfc.to_csv(out_path, index=False, date_format='%Y-%m-%d %H:%M:%S')

    # compute and pass separate CSV with weather model variance
    out_path = out_path.with_name(
        f"{out_path.stem}_WM_variance{out_path.suffix}"
    )
    # filter out obs by error
    if obs_errlimit != float('inf'):
        prefilt_len = len(dfc)
        dfc = dfc[dfc["sigZTD"] <= obs_errlimit]
        filt_len = len(dfc)
        errlimit_dropped = prefilt_len - filt_len
        logger.warning(
            "Dropped %d observations with sigZTD > input %s (%d remaining).",
            errlimit_dropped,
            obs_errlimit,
            filt_len,
        )

    # get temporal sampling stats
    mean_delta_days, mode_delta_days = sampling_delta_stats(dfc)

    logger.info(
        "Global mean delta (days): %s "
        "Global mode delta (days): %s",
        mean_delta_days,
        mode_delta_days,
    )

    # determine coverage window across all retained observations
    df_date = dfc["Datetime"].dt.normalize()
    global_start = df_date.min()
    global_end = df_date.max()
    if pd.isna(global_start) or pd.isna(global_end):
        n_global_days = 0
    else:
        n_global_days_total_span = (global_end - global_start).days + 1
        # capture reference, maximum temporal sampling
        # using mean time delta of all observations
        n_global_days = math.ceil(n_global_days_total_span  / mode_delta_days)
        logger.info(
            "The earliest/latest dates found are %s & %s which spans %d days, "
            "with a sampling mode of %d days",
            global_start,
            global_end,
            n_global_days_total_span,
            n_global_days,
        )

    dfc_qm = (
        dfc.groupby("ID", dropna=False, sort=True)
        .apply(
            variance_analysis,
            allow_nan_for_negative=allow_nan_for_negative,
            has_localtime=has_localtime,
            global_start=global_start,
            global_end=global_end,
            n_global_days=n_global_days,
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    # compute statistics for flagged `sigma_model_neg
    dfc_qm.drop_duplicates(inplace=True)
    n_before = len(dfc_qm)
    sigma_model_filt_len = dfc_qm["sigma_model_neg"].sum()
    if sigma_model_filt_len > 0:
        df_neg = dfc_qm.loc[dfc_qm["sigma_model_neg"]]
        stats = df_neg[
            ["n_unique_days", "pct_days_global"]
        ].agg(["mean", "median", "min", "max"])

        logger.warning(
            "Negative σ_model² for %d stations. "
            "Unique days sampled: mean=%.2f, median=%.2f, "
            "min=%.2f, max=%.2f. "
            "Daily overlap: mean=%.2f%%, median=%.2f%%, "
            "min=%.2f%%, max=%.2f%%.",
            len(df_neg),
            stats.loc["mean", "n_unique_days"],
            stats.loc["median", "n_unique_days"],
            stats.loc["min", "n_unique_days"],
            stats.loc["max", "n_unique_days"],
            stats.loc["mean", "pct_days_global"],
            stats.loc["median", "pct_days_global"],
            stats.loc["min", "pct_days_global"],
            stats.loc["max", "pct_days_global"],
        )

    # Drop all lines with NaNs and duplicates
    dfc_qm = dfc_qm.drop(columns=["sigma_model_neg"])
    dfc_qm.dropna(how="any", inplace=True)
    nan_filt_len = n_before - len(dfc_qm)

    if min_pct_days > 0:
        before_filter = len(dfc_qm)
        dfc_qm = dfc_qm.loc[dfc_qm["pct_days_global"] > min_pct_days]
        removed = before_filter - len(dfc_qm)
        logger.warning(
            "Dropped %s station(s) with pct_days_global <= %.2f "
            "(%s retained).",
            removed,
            min_pct_days,
            len(dfc_qm),
        )

    if not allow_nan_for_negative:
        n_flagged = (dfc_qm["sigma_model"] == 0).sum()
        logger.warning(
            "%d/%d stations contain 0 sigma values.",
            n_flagged,
            len(dfc_qm),
        )

    if nan_filt_len > 0:
        logger.warning(
            "Dropped %d stations containing NaN due to all imposed filters "
            "(see other warnings above for more details), "
            "(%d) stations remaining.",
            nan_filt_len,
            len(dfc_qm),
        )

    # capture final stats
    stats = dfc_qm[
        ["n_unique_days", "pct_days_global"]
    ].agg(["mean", "median", "min", "max"])

    logger.info(
        "%d kept stations. "
        "Unique days sampled: mean=%.2f, median=%.2f, "
        "min=%.2f, max=%.2f. "
        "Daily overlap: mean=%.2f%%, median=%.2f%%, "
        "min=%.2f%%, max=%.2f%%.",
        len(dfc_qm),
        stats.loc["mean", "n_unique_days"],
        stats.loc["median", "n_unique_days"],
        stats.loc["min", "n_unique_days"],
        stats.loc["max", "n_unique_days"],
        stats.loc["mean", "pct_days_global"],
        stats.loc["median", "pct_days_global"],
        stats.loc["min", "pct_days_global"],
        stats.loc["max", "pct_days_global"],
    )

    dfc_qm.to_csv(
        out_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.6f",
    )
    del dfc_qm
