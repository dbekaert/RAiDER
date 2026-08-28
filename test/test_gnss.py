import datetime
import os
from pathlib import Path

import pandas as pd
import pytest

from RAiDER.gnss.downloadGNSSDelays import download_tropo_delays, filterToBBox, get_station_list, get_stats_by_llh
from RAiDER.gnss.processDelayFiles import (
    addDateTimeToFiles,
    concatDelayFiles,
    getDateTime,
    readZTDFile,
)
from RAiDER.gnss.processDelayFiles import main as process_main
from RAiDER.models.customExceptions import NoStationDataFoundError
from test import TEST_DIR, pushd


SCENARIO2_DIR = os.path.join(TEST_DIR, "scenario_2")


def file_len(path: Path) -> int:
    with path.open('rb') as f:
        return sum(1 for _ in f)


@pytest.fixture
def temp_file():
    df = pd.DataFrame(
        {
            'ID': ['STAT1', 'STAT2', 'STAT3'],
            'Lat': [15.0, 20., 25.0],
            'Lon': [-100, -90., -85.],
            'totalDelay': [1., 1.5, 2.],
        }
    )
    return df


def test_getDateTime():
    f1 = Path('20080101T060000')
    f2 = Path('20080101T560000')
    f3 = Path('20080101T0600000')
    f4 = Path('20080101_060000')
    f5 = Path('2008-01-01T06:00:00')
    assert getDateTime(f1) == datetime.datetime(2008, 1, 1, 6, 0, 0)
    with pytest.raises(ValueError):
        getDateTime(f2)
    assert getDateTime(f3) == datetime.datetime(2008, 1, 1, 6, 0, 0)
    with pytest.raises(AttributeError):
        getDateTime(f4)
    with pytest.raises(AttributeError):
        getDateTime(f5)


def test_addDateTimeToFiles1(tmp_path, temp_file):
    df = temp_file

    with pushd(tmp_path):
        new_path = tmp_path / 'tmp.csv'
        df.to_csv(new_path, index=False)
        addDateTimeToFiles([new_path])
        df = pd.read_csv(new_path)
        assert 'Datetime' not in df.columns


def test_addDateTimeToFiles2(tmp_path, temp_file):
    f1 = '20080101T060000'
    df = temp_file

    with pushd(tmp_path):
        new_path = tmp_path / f'tmp{f1}.csv'
        df.to_csv(new_path, index=False)
        addDateTimeToFiles([new_path])
        df = pd.read_csv(new_path)
        assert 'Datetime' in df.columns


def test_concatDelayFiles(tmp_path, temp_file):
    f1 = '20080101T060000'
    df = temp_file

    with pushd(tmp_path):
        new_path1 = tmp_path / f'tmp{f1}_1.csv'
        new_path2 = tmp_path / f'tmp{f1}_2.csv'
        df.to_csv(new_path1, index=False)
        df.to_csv(new_path2, index=False)
        file_length = file_len(new_path1)
        addDateTimeToFiles([new_path1, new_path2])

        out_path = tmp_path / 'out.csv'
        concatDelayFiles(
            [new_path1, new_path2],
            outName=out_path
        )
    assert file_len(out_path) == file_length


def test_get_stats_by_llh2():
    stations = get_stats_by_llh(llhBox=[10, 18, -93, -88])
    assert isinstance(stations, pd.DataFrame)


def test_get_stats_by_llh3():
    with pytest.raises(ValueError):
        get_stats_by_llh(llhBox=[10, 18, 360-93, 360-88])


def test_get_station_list():
    stations, output_file = get_station_list(stationFile=os.path.join(
        SCENARIO2_DIR, 'stations.csv'), writeStationFile=False)
    assert isinstance(stations, list)
    assert isinstance(output_file, pd.DataFrame)


def test_download_tropo_delays1():
    with pytest.raises(NotImplementedError):
        download_tropo_delays(stats=['GUAT', 'SLAC', 'CRSE'], years=[
                              2022], gps_repo='dummy_repo')


def test_download_tropo_delays2():
    with pytest.raises(NoStationDataFoundError):
        download_tropo_delays(stats=['dummy_station'], years=[2022])


def test_download_tropo_delays2(tmp_path):
    with pushd(tmp_path):
        stations, output_file = get_station_list(
            stationFile=os.path.join(SCENARIO2_DIR, 'stations.csv')
        )

        # spot check a couple of stations
        assert 'CAPE' in stations
        assert 'FGNW' in stations
        assert isinstance(output_file, str)

        # try downloading the delays
        download_tropo_delays(stats=stations, years=[2022], writeDir=tmp_path)
        assert True


def test_filterByBBox1():
    _, station_data = get_station_list(stationFile=os.path.join(
        SCENARIO2_DIR, 'stations.csv'), writeStationFile=False)
    with pytest.raises(ValueError):
        filterToBBox(station_data, llhBox=[34, 38, 240, 245])


def test_filterByBBox2():
    _, station_data = get_station_list(stationFile=os.path.join(
        SCENARIO2_DIR, 'stations.csv'), writeStationFile=False)
    new_data = filterToBBox(station_data, llhBox=[34, 38, -120, -115])
    for stat in ['CAPE', 'MHMS', 'NVCO']:
        assert stat not in new_data['ID'].to_list()
    for stat in ['FGNW', 'JPLT', 'NVTP', 'WLHG', 'WORG']:
        assert stat in new_data['ID'].to_list()


def write_ztd_csv(path: Path, **columns) -> Path:
    """Write a GNSS ZTD file with whatever columns a test needs."""
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


def test_readZTDFile_date_and_times(tmp_path):
    """'Date' plus a 'times' seconds-of-day column combine into 'Datetime'."""
    f = write_ztd_csv(
        tmp_path / 'UNRcombinedGPS_ztd.csv',
        ID=['STAT1', 'STAT2'],
        Date=['2020-01-30', '2020-01-30'],
        times=[0, 45296],  # midnight and 12:34:56
        ZTD=[2.3, 2.4],
    )
    data = readZTDFile(f)
    assert data['Datetime'].to_list() == [
        datetime.datetime(2020, 1, 30, 0, 0, 0),
        datetime.datetime(2020, 1, 30, 12, 34, 56),
    ]


def test_readZTDFile_date_without_times(tmp_path):
    """Without a 'times' column every observation lands at midnight."""
    f = write_ztd_csv(
        tmp_path / 'UNRcombinedGPS_ztd.csv',
        ID=['STAT1', 'STAT2'],
        Date=['2020-01-30', '2020-01-31'],
        ZTD=[2.3, 2.4],
    )
    data = readZTDFile(f)
    assert data['Datetime'].to_list() == [
        datetime.datetime(2020, 1, 30),
        datetime.datetime(2020, 1, 31),
    ]


def test_readZTDFile_unparseable_times_are_treated_as_midnight(tmp_path):
    """A junk 'times' value is coerced to zero rather than failing the read."""
    f = write_ztd_csv(
        tmp_path / 'UNRcombinedGPS_ztd.csv',
        ID=['STAT1', 'STAT2'],
        Date=['2020-01-30', '2020-01-30'],
        times=['not-a-number', 3600],
        ZTD=[2.3, 2.4],
    )
    data = readZTDFile(f)
    assert data['Datetime'].to_list() == [
        datetime.datetime(2020, 1, 30, 0, 0, 0),
        datetime.datetime(2020, 1, 30, 1, 0, 0),
    ]


def test_readZTDFile_existing_datetime_column(tmp_path):
    """An existing 'Datetime' column is used as-is when there is no 'Date'."""
    f = write_ztd_csv(
        tmp_path / 'UNRcombinedGPS_ztd.csv',
        ID=['STAT1'],
        Datetime=['2020-01-30 13:52:45'],
        ZTD=[2.3],
    )
    data = readZTDFile(f)
    assert data['Datetime'].to_list() == [datetime.datetime(2020, 1, 30, 13, 52, 45)]


def test_readZTDFile_datetime_from_filename(tmp_path, caplog):
    """With no time column at all, the filename timestamp is used, with a warning."""
    f = write_ztd_csv(
        tmp_path / 'ERA5_Delay_20210308T120000_ztd.csv',
        ID=['STAT1', 'STAT2'],
        ZTD=[2.3, 2.4],
    )
    data = readZTDFile(f)

    assert data['Datetime'].to_list() == [datetime.datetime(2021, 3, 8, 12, 0, 0)] * 2
    # Every row gets the same stamp, so the user has to be told.
    assert 'no longer distinguishable in time' in caplog.text


def test_readZTDFile_raises_without_any_time_information(tmp_path):
    """No time column and no timestamp in the filename is an error, not a guess."""
    f = write_ztd_csv(
        tmp_path / 'stations.csv',
        ID=['STAT1'],
        Lat=[15.0],
        Lon=[-100.0],
        ZTD=[2.3],
    )
    with pytest.raises(ValueError, match='cannot be added automatically'):
        readZTDFile(f)


def test_readZTDFile_renames_the_delay_column(tmp_path):
    """The delay column named by col_name is renamed to 'ZTD'."""
    f = write_ztd_csv(
        tmp_path / 'UNRcombinedGPS_ztd.csv',
        ID=['STAT1'],
        Date=['2020-01-30'],
        totalDelay=[2.3],
    )
    data = readZTDFile(f, col_name='totalDelay')
    assert 'totalDelay' not in data.columns
    assert data['ZTD'].to_list() == [2.3]


def write_delay_pair(directory: Path, ztd_days, raider_days, with_coords=True):
    """Write a matched RAiDER/GNSS delay file pair for the combine workflow.

    The GNSS file optionally omits Lat/Lon/Hgt_m, as UNR per-station delay
    files do, so that the backfill from the RAiDER file can be exercised.
    """
    stations = ['STA1', 'STA2']
    raider_rows, ztd_rows = [], []
    for station in stations:
        for k, day in enumerate(raider_days):
            raider_rows.append({
                'ID': station, 'Lat': 34.0, 'Lon': -118.0, 'Hgt_m': 100.0,
                'Datetime': day + pd.Timedelta(hours=12),
                'wetDelay': 0.1, 'hydroDelay': 2.2, 'totalDelay': 2.3 + 0.01 * k,
            })
        for k, day in enumerate(ztd_days):
            # Residuals alternate in sign so the per-station variance analysis
            # has something to work with rather than a constant offset.
            ztd_rows.append({
                'ID': station, 'Datetime': day + pd.Timedelta(hours=12),
                'ZTD': 2.3 + 0.01 * k + (0.02 if k % 2 else -0.02),
                'sigZTD': 0.005, 'times': 43200,
            })

    ztd_frame = pd.DataFrame(ztd_rows)
    if with_coords:
        ztd_frame['Lat'] = 34.0
        ztd_frame['Lon'] = -118.0
        ztd_frame['Hgt_m'] = 100.0

    raider_file = directory / 'raider_delays.csv'
    ztd_file = directory / 'gnss_delays.csv'
    pd.DataFrame(raider_rows).to_csv(raider_file, index=False)
    ztd_frame.to_csv(ztd_file, index=False)
    return raider_file, ztd_file


def test_main_backfills_station_coordinates_from_raider_file(tmp_path):
    """A GNSS file with no station coordinates inherits them from the RAiDER file."""
    days = pd.date_range('2020-01-01', periods=12, freq='D')
    raider_file, ztd_file = write_delay_pair(tmp_path, days, days, with_coords=False)
    assert 'Lat' not in pd.read_csv(ztd_file).columns

    out_path = tmp_path / 'combined.csv'
    process_main(raider_file, ztd_file, out_path=out_path)

    combined = pd.read_csv(out_path)
    assert combined['Lat'].dropna().unique().tolist() == [34.0]
    assert combined['Lon'].dropna().unique().tolist() == [-118.0]


def test_main_raises_when_files_share_no_observations(tmp_path):
    """Non-overlapping dates fail with an explanation, not a later KeyError."""
    raider_days = pd.date_range('2020-01-01', periods=12, freq='D')
    ztd_days = pd.date_range('2021-06-01', periods=12, freq='D')
    raider_file, ztd_file = write_delay_pair(tmp_path, ztd_days, raider_days)

    with pytest.raises(ValueError, match='No common observations'):
        process_main(raider_file, ztd_file, out_path=tmp_path / 'combined.csv')
