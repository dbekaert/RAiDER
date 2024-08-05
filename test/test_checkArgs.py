import datetime
import os
import shutil
import pytest

import multiprocessing as mp
import numpy as np
import pandas as pd

from test import TEST_DIR, pushd

from RAiDER.cli import DEFAULT_DICT
from RAiDER.checkArgs import checkArgs, makeDelayFileNames, get_raster_ext
from RAiDER.llreader import BoundingBox, StationFile, RasterRDR
from RAiDER.losreader import Zenith, Conventional, Raytracing
from RAiDER.models.gmao import GMAO


SCENARIO_1 = os.path.join(TEST_DIR, "scenario_1")
SCENARIO_2 = os.path.join(TEST_DIR, "scenario_2")


@pytest.fixture(autouse=True)
def args():
    d = DEFAULT_DICT
    d["date_list"] = [datetime.datetime(2018, 1, 1)]
    d["time"] = datetime.time(12, 0, 0)
    d["aoi"] = BoundingBox([38, 39, -92, -91])
    d["los"] = Zenith()
    d["weather_model"] = GMAO()

    for f in "weather_files weather_dir".split():
        shutil.rmtree(f) if os.path.exists(f) else ""
    return d


def isWriteable(dirpath):
    """Test whether a directory is writeable"""
    try:
        filehandle = open(os.path.join(dirpath, "tmp.txt"), "w")
        filehandle.close()
        return True
    except IOError:
        return False


def test_checkArgs_outfmt_1(args):
    """Test that passing height levels with hdf5 outformat works"""
    args = args
    args.file_format = "h5"
    args.heightlvls = [10, 100, 1000]
    checkArgs(args)
    assert os.path.splitext(args.wetFilenames[0])[-1] == ".h5"


def test_checkArgs_outfmt_2(args):
    """Test that passing a raster format with height levels throws an error"""
    args = args
    args.heightlvs = [10, 100, 1000]
    args.file_format = "GTiff"
    args = checkArgs(args)
    assert os.path.splitext(args.wetFilenames[0])[-1] == ".nc"


def test_checkArgs_outfmt_3(args):
    """Test that passing a raster format with height levels throws an error"""
    args = args
    with pytest.raises(FileNotFoundError):
        args.aoi = StationFile(os.path.join("fake_dir", "stations.csv"))


def test_checkArgs_outfmt_4(args):
    """Test that passing a raster format with height levels throws an error"""
    args = args
    args.aoi = RasterRDR(
        lat_file=os.path.join(SCENARIO_1, "geom", "lat.dat"),
        lon_file=os.path.join(SCENARIO_1, "geom", "lon.dat"),
    )
    argDict = checkArgs(args)
    assert argDict.aoi.type() == "radar_rasters"


def test_checkArgs_outfmt_5(args, tmp_path):
    with pushd(tmp_path):
        args = args
        args.aoi = StationFile(os.path.join(SCENARIO_2, "stations.csv"))
        argDict = checkArgs(args)
        assert pd.read_csv(argDict["wetFilenames"][0]).shape == (8, 4)


def test_checkArgs_outloc_1(args):
    """Test that the default output and weather model directories are correct"""
    args = args
    argDict = checkArgs(args)
    out = argDict["output_directory"]
    wmLoc = argDict["weather_model_directory"]
    assert os.path.abspath(out) == os.getcwd()
    assert os.path.abspath(wmLoc) == os.path.join(os.getcwd(), "weather_files")


def test_checkArgs_outloc_2(args, tmp_path):
    """Tests that the correct output location gets assigned when provided"""
    with pushd(tmp_path):
        args = args
        args.output_directory = tmp_path
        argDict = checkArgs(args)
        out = argDict["output_directory"]
        assert out == tmp_path


def test_checkArgs_outloc_2b(args, tmp_path):
    """Tests that the weather model directory gets passed through by itself"""
    with pushd(tmp_path):
        args = args
        args.output_directory = tmp_path
        args.weather_model_directory = "weather_dir"
        argDict = checkArgs(args)
        assert argDict["weather_model_directory"] == "weather_dir"


def test_checkArgs_outloc_3(args, tmp_path):
    """Tests that the weather model directory gets created when needed"""
    with pushd(tmp_path):
        args = args
        args.output_directory = tmp_path
        argDict = checkArgs(args)
        assert os.path.isdir(argDict["weather_model_directory"])


def test_checkArgs_outloc_4(args):
    """Tests for creating writeable weather model directory"""
    args = args
    argDict = checkArgs(args)

    assert isWriteable(argDict["weather_model_directory"])


def test_filenames_1(args):
    """tests that the correct filenames are generated"""
    args = args
    argDict = checkArgs(args)
    assert "Delay" not in argDict["wetFilenames"][0]
    assert "wet" in argDict["wetFilenames"][0]
    assert "hydro" in argDict["hydroFilenames"][0]
    assert "20180101" in argDict["wetFilenames"][0]
    assert "20180101" in argDict["hydroFilenames"][0]
    assert len(argDict["hydroFilenames"]) == 1


def test_filenames_2(args):
    """tests that the correct filenames are generated"""
    args = args
    args["output_directory"] = SCENARIO_2
    args.aoi = StationFile(os.path.join(SCENARIO_2, "stations.csv"))
    argDict = checkArgs(args)
    assert "20180101" in argDict["wetFilenames"][0]
    assert len(argDict["wetFilenames"]) == 1


def test_makeDelayFileNames_1():
    assert makeDelayFileNames(None, None, "h5", "name", "dir") == (
        "dir/name_wet_ztd.h5",
        "dir/name_hydro_ztd.h5",
    )


def test_makeDelayFileNames_2():
    assert makeDelayFileNames(None, (), "h5", "name", "dir") == (
        "dir/name_wet_std.h5",
        "dir/name_hydro_std.h5",
    )


def test_makeDelayFileNames_3():
    assert makeDelayFileNames(
        datetime.datetime(2020, 1, 1, 1, 2, 3), None, "h5", "model_name", "dir"
    ) == (
        "dir/model_name_wet_20200101T010203_ztd.h5",
        "dir/model_name_hydro_20200101T010203_ztd.h5",
    )


def test_makeDelayFileNames_4():
    assert makeDelayFileNames(
        datetime.datetime(1900, 12, 31, 1, 2, 3), "los", "h5", "model_name", "dir"
    ) == (
        "dir/model_name_wet_19001231T010203_std.h5",
        "dir/model_name_hydro_19001231T010203_std.h5",
    )


def test_get_raster_ext():
    with pytest.raises(ValueError):
        get_raster_ext("dummy_format")
