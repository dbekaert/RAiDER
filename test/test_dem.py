import os
import pytest

from test import TEST_DIR, pushd
from RAiDER.dem import download_dem


def test_download_dem_1():
    SCENARIO_1 = TEST_DIR / "scenario_4"
    hts, meta = download_dem(
        dem_path=SCENARIO_1 / 'warpedDEM.dem', 
        overwrite=False
    )
    assert hts.shape == (45,226)
    assert meta['crs'] is None


def test_download_dem_2():
    with pytest.raises(ValueError):
        download_dem()


def test_download_dem_3(tmp_path):
    with pushd(tmp_path):
        fname = os.path.join(tmp_path, 'tmp_file.nc')
        with pytest.raises(ValueError):
            download_dem(dem_path=fname)


@pytest.mark.long
def test_download_dem_4(tmp_path):
    with pushd(tmp_path):
        fname = os.path.join(tmp_path, 'tmp_file.nc')
        z,m = download_dem(dem_path=fname, overwrite=True, ll_bounds=[37.9,38.,-91.8,-91.7], writeDEM=True)
        assert len(z.shape) == 2
        assert 'crs' in m.keys()


