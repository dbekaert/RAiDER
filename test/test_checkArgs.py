import datetime
from pathlib import Path

import pandas as pd
import pytest

from RAiDER.checkArgs import checkArgs, get_raster_ext, makeDelayFileNames
from RAiDER.cli.types import AOIGroup, DateGroup, HeightGroupUnparsed, LOSGroup, RunConfig, RuntimeGroup, TimeGroup
from RAiDER.llreader import BoundingBox, RasterRDR, StationFile
from RAiDER.losreader import Conventional, Zenith
from RAiDER.models.gmao import GMAO
from test import TEST_DIR


SCENARIO_1 = TEST_DIR / 'scenario_1'
SCENARIO_2 = TEST_DIR / 'scenario_2'


@pytest.fixture()
def args_default_out_dir() -> RunConfig:
    return RunConfig(
        weather_model=GMAO(),
        date_group=DateGroup(date_list=[datetime.datetime(2018, 1, 1)]),
        time_group=TimeGroup(time=datetime.time(12, 0, 0)),
        aoi_group=AOIGroup(aoi=BoundingBox((38, 39, -92, -91))),
        los_group=LOSGroup(los=Zenith()),
        height_group=HeightGroupUnparsed(),
        runtime_group=RuntimeGroup(),
    )


@pytest.fixture()
def args(tmp_path: Path, args_default_out_dir: RunConfig) -> RunConfig:
    args_default_out_dir.runtime_group = RuntimeGroup(output_directory=tmp_path)
    return args_default_out_dir


def isWriteable(dirpath: Path) -> bool:
    """Test whether a directory is writeable."""
    try:
        with (dirpath / 'tmp.txt').open('w'):
            pass
        return True
    except OSError:
        return False


def test_checkArgs_outfmt_1(args: RunConfig) -> None:
    args.runtime_group.file_format = 'h5'
    args.height_group.height_levels = [10, 100, 1000]
    args = checkArgs(args)
    assert Path(args.wetFilenames[0]).suffix == '.h5'


def test_checkArgs_outfmt_2(args: RunConfig) -> None:
    args.runtime_group.file_format = 'GTiff'
    args.height_group.height_levels = [10, 100, 1000]
    args = checkArgs(args)
    assert Path(args.wetFilenames[0]).suffix == '.nc'


def test_checkArgs_outfmt_3(args: RunConfig) -> None:
    with pytest.raises(FileNotFoundError):
        args.aoi_group.aoi = StationFile(Path('fake_dir/stations.csv'))


def test_checkArgs_outfmt_4(args: RunConfig) -> None:
    args.aoi_group.aoi = RasterRDR(
        lat_file=SCENARIO_1 / 'geom/lat.dat',
        lon_file=SCENARIO_1 / 'geom/lon.dat',
    )
    args = checkArgs(args)
    assert args.aoi_group.aoi.type() == 'radar_rasters'


def test_checkArgs_outfmt_5(args: RunConfig) -> None:
    args.aoi_group.aoi = StationFile(SCENARIO_2 / 'stations.csv')
    args = checkArgs(args)
    assert pd.read_csv(args.wetFilenames[0]).shape == (8, 4)


def test_checkArgs_outloc_1(args_default_out_dir: RunConfig) -> None:
    """Test that the default output and weather model directories are correct."""
    cwd = Path.cwd().resolve()
    expected_wm_dir = (cwd / 'weather_files')
    assert not expected_wm_dir.exists(), (
        'weather_files/ already exists; cannot ensure weather RAiDER will create it. '
        'Please remove this directory and run the test again.'
    )
    run_config = checkArgs(args_default_out_dir)
    actual_out_dir = run_config.runtime_group.output_directory
    actual_wm_dir = run_config.runtime_group.weather_model_directory
    actual_wm_dir.rmdir()
    assert actual_out_dir.resolve() == cwd
    assert actual_wm_dir.resolve() == expected_wm_dir


def test_checkArgs_outloc_2(tmp_path: Path, args_default_out_dir: RunConfig) -> None:
    """Tests that the correct output location gets assigned when provided."""
    args_default_out_dir.runtime_group = RuntimeGroup(output_directory=tmp_path)
    argDict = checkArgs(args_default_out_dir)
    out = argDict.runtime_group.output_directory
    assert out.resolve() == tmp_path.resolve()


def test_checkArgs_outloc_2b(tmp_path: Path, args: RunConfig) -> None:
    """Tests that the weather model directory gets passed through by itself."""
    wm_dir = tmp_path / 'weather_dir'
    args.runtime_group.weather_model_directory = wm_dir
    argDict = checkArgs(args)
    assert argDict.runtime_group.weather_model_directory == wm_dir


def test_checkArgs_outloc_3(args: RunConfig) -> None:
    """Tests that the weather model directory gets created when needed."""
    argDict = checkArgs(args)
    assert argDict.runtime_group.weather_model_directory.is_dir()


def test_checkArgs_outloc_4(args: RunConfig) -> None:
    """Tests for creating writeable weather model directory."""
    argDict = checkArgs(args)
    assert isWriteable(argDict.runtime_group.weather_model_directory)


def test_filenames_1(args: RunConfig) -> None:
    """tests that the correct filenames are generated."""
    argDict = checkArgs(args)
    assert 'Delay' not in argDict.wetFilenames[0]
    assert 'wet' in argDict.wetFilenames[0]
    assert 'hydro' in argDict.hydroFilenames[0]
    assert '20180101' in argDict.wetFilenames[0]
    assert '20180101' in argDict.hydroFilenames[0]
    assert len(argDict.hydroFilenames) == 1


def test_filenames_2(args: RunConfig) -> None:
    """Tests that the correct filenames are generated."""
    args.aoi_group.aoi = StationFile(SCENARIO_2 / 'stations.csv')
    argDict = checkArgs(args)
    assert '20180101' in argDict.wetFilenames[0]
    assert len(argDict.wetFilenames) == 1


def test_makeDelayFileNames_1() -> None:
    assert makeDelayFileNames(None, None, 'h5', 'name', Path('dir')) == ('dir/name_wet_ztd.h5', 'dir/name_hydro_ztd.h5')


def test_makeDelayFileNames_2() -> None:
    assert makeDelayFileNames(None, Conventional(), 'h5', 'name', Path('dir')) == (
        'dir/name_wet_std.h5',
        'dir/name_hydro_std.h5',
    )


def test_makeDelayFileNames_3() -> None:
    assert makeDelayFileNames(datetime.datetime(2020, 1, 1, 1, 2, 3), None, 'h5', 'model_name', Path('dir')) == (
        'dir/model_name_wet_20200101T010203_ztd.h5',
        'dir/model_name_hydro_20200101T010203_ztd.h5',
    )


def test_makeDelayFileNames_4() -> None:
    assert makeDelayFileNames(
        datetime.datetime(1900, 12, 31, 1, 2, 3), Conventional(), 'h5', 'model_name', Path('dir')
    ) == (
        'dir/model_name_wet_19001231T010203_std.h5',
        'dir/model_name_hydro_19001231T010203_std.h5',
    )


def test_get_raster_ext() -> None:
    with pytest.raises(ValueError):
        get_raster_ext('dummy_format')
