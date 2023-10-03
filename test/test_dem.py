import os
import pytest

from test import TEST_DIR

from RAiDER.dem import download_dem


def test_download_dem_1():
    SCENARIO_1 = os.path.join(TEST_DIR, "scenario_4")
    hts, meta = download_dem(
        outName=os.path.join(SCENARIO_1,'warpedDEM.dem'), 
        overwrite=False
    )
    assert hts.shape == (45,226)
    assert meta['crs'] is None


