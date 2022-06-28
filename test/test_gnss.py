import datetime
import os
import pytest

import pandas as pd

from test import pushd

from RAiDER.gnss.processDelayFiles import (
    addDateTimeToFiles,
    getDateTime,
    concatDelayFiles
)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


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
    f1 = '20080101T060000'
    f2 = '20080101T560000'
    f3 = '20080101T0600000'
    f4 = '20080101_060000'
    f5 = '2008-01-01T06:00:00'
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
        new_name = os.path.join(tmp_path, 'tmp.csv')
        df.to_csv(new_name, index=False)
        addDateTimeToFiles([new_name])
        df = pd.read_csv(new_name)
        assert 'Datetime' not in df.columns


def test_addDateTimeToFiles2(tmp_path, temp_file):
    f1 = '20080101T060000'
    df = temp_file

    with pushd(tmp_path):
        new_name = os.path.join(
            tmp_path,
            'tmp' + f1 + '.csv'
        )
        df.to_csv(new_name, index=False)
        addDateTimeToFiles([new_name])
        df = pd.read_csv(new_name)
        assert 'Datetime' in df.columns


def test_concatDelayFiles(tmp_path, temp_file):
    f1 = '20080101T060000'
    df = temp_file

    with pushd(tmp_path):
        new_name = os.path.join(
            tmp_path,
            'tmp' + f1 + '.csv'
        )
        new_name2 = os.path.join(
            tmp_path,
            'tmp' + f1 + '_2.csv'
        )
        df.to_csv(new_name, index=False)
        df.to_csv(new_name2, index=False)
        file_length = file_len(new_name)
        addDateTimeToFiles([new_name, new_name2])

        out_name = os.path.join(tmp_path, 'out.csv')
        concatDelayFiles(
            [new_name, new_name2],
            out_name=out_name
        )
    assert file_len(out_name) == file_length
