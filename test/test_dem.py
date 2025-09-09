from pathlib import Path

import pytest

from RAiDER.dem import download_dem
from test import TEST_DIR


def test_download_dem_1() -> None:
    SCENARIO_1 = TEST_DIR / 'scenario_4'
    hts, meta = download_dem(dem_path=SCENARIO_1 / 'warpedDEM.rdr', overwrite=False)
    assert hts.shape == (45, 226)
    assert meta is not None
    assert meta['crs'] is None


def test_download_dem_2() -> None:
    with pytest.raises(ValueError):
        download_dem()


def test_download_dem_3(tmp_path: Path) -> None:
    path = tmp_path / 'tmp_file.nc'
    with pytest.raises(ValueError):
        download_dem(dem_path=path)


@pytest.mark.long
def test_download_dem_4(tmp_path: Path) -> None:
    path = tmp_path / 'tmp_file.nc'
    z, m = download_dem(dem_path=path, overwrite=True, ll_bounds=[37.9, 38.0, -91.8, -91.7], writeDEM=True)
    assert len(z.shape) == 2
    assert m is not None
    assert 'crs' in m.keys()
