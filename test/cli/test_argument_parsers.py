import pytest

from datetime import date, time

import RAiDER.runProgram

from RAiDER.rays import ZenithLVGenerator


@pytest.fixture
def delay_parser():
    return RAiDER.runProgram.create_parser()


@pytest.fixture
def stats_parser():
    pass


@pytest.fixture
def gnss_parser():
    pass


def test_delay_args(delay_parser):
    args = delay_parser.parse_args([
        '--date', '20200103',
        '--time', '23:00:00',
        '--latlon', 'latfile.dat', 'lonfile.dat',
        '--model', 'ERA5',
        '--zref', '20000',
        '-v',
        '--out', 'test/scenario_1/'
    ])

    assert args.dateList == [date(2020, 1, 3)]
    assert args.time == time(23, 0, 0)
    assert args.latlon == ['latfile.dat', 'lonfile.dat']
    assert args.bbox is None
    assert args.station_file is None
    assert args.lineofsight is ZenithLVGenerator
    assert args.dem is None
    assert args.heightlvs is None
    assert args.model == "ERA5"
    assert args.files is None
    assert args.wmLoc is None
    assert args.zref == 20000.0
    assert args.outformat is None
    assert args.out == 'test/scenario_1/'
    assert args.download_only is False
    assert args.verbose == 1


@pytest.mark.xfail(reason='Have to update test to handle new LOS generator objects')
def test_delay_los_mutually_exclusive(delay_parser):
    with pytest.raises(SystemExit):
        delay_parser.parse_args([
            '--date', '20200103',
            '--time', '23:00:00',
            '--lineofsight', 'losfile',
        ])


def test_delay_aoi_mutually_exclusive(delay_parser):
    with pytest.raises(SystemExit):
        delay_parser.parse_args([
            '--date', '20200103',
            '--time', '23:00:00',
            '--bbox', '10', '20', '30', '40',
            '--latlon', 'lat', 'lon',
            '--station_file', 'station_file'
        ])

    with pytest.raises(SystemExit):
        delay_parser.parse_args([
            '--date', '20200103',
            '--time', '23:00:00',
            '--bbox', '10', '20', '30', '40',
            '--latlon', 'lat', 'lon',
        ])

    with pytest.raises(SystemExit):
        delay_parser.parse_args([
            '--date', '20200103',
            '--time', '23:00:00',
            '--bbox', '10', '20', '30', '40',
            '--station_file', 'station_file'
        ])

    with pytest.raises(SystemExit):
        delay_parser.parse_args([
            '--date', '20200103',
            '--time', '23:00:00',
            '--latlon', 'lat', 'lon',
            '--station_file', 'station_file'
        ])

    # AOI is required
    with pytest.raises(SystemExit):
        delay_parser.parse_args([
            '--date', '20200103',
            '--time', '23:00:00',
        ])


def test_delay_model(delay_parser):
    with pytest.raises(SystemExit):
        delay_parser.parse_args([
            '--date', '20200103',
            '--time', '23:00:00',
            '--station_file', 'station_file',
            '--model', 'FOOBAR'
        ])

    args = delay_parser.parse_args([
        '--date', '20200103',
        '--time', '23:00:00',
        '--station_file', 'station_file',
        '--model', 'era-5'
    ])
    assert args.model == "ERA5"
