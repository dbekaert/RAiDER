import os
import pytest

from test import TEST_DIR

from RAiDER.dem import download_dem


def test_download_dem_1():
    SCENARIO_1 = os.path.join(TEST_DIR, "scenario_1")
    hts = download_dem(outName=os.path.join(SCENARIO_1,'geom', 'hgt.rdr'), overwrite=False)
    assert True


