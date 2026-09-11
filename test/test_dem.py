import logging

import pytest

import numpy as np
import rasterio
from affine import Affine

from test import TEST_DIR, pushd
from RAiDER.dem import DEM_DATUM, DEM_DATUM_TAG, download_dem


def test_download_dem_1():
    SCENARIO_1 = TEST_DIR / 'scenario_4'
    hts, meta = download_dem(dem_path=SCENARIO_1 / 'warpedDEM.rdr', overwrite=False)
    assert hts.shape == (45, 226)
    assert meta is not None
    assert meta['crs'] is None


def test_download_dem_2():
    with pytest.raises(ValueError):
        download_dem()


def test_download_dem_3(tmp_path):
    with pushd(tmp_path):
        path = tmp_path / 'tmp_file.nc'
        with pytest.raises(ValueError):
            download_dem(dem_path=path)


@pytest.mark.long
def test_download_dem_4(tmp_path):
    with pushd(tmp_path):
        path = tmp_path / 'tmp_file.nc'
        z, m = download_dem(dem_path=path, overwrite=True, ll_bounds=[37.9, 38.0, -91.8, -91.7], writeDEM=True)
        assert len(z.shape) == 2
        assert m is not None
        assert 'crs' in m.keys()


# ---------------------------------------------------------------------------
# dem_stitcher integration
# ---------------------------------------------------------------------------


@pytest.mark.long
def test_stitch_dem_live_smoke():
    """Exercise the real dem_stitcher with the exact arguments download_dem uses.

    A 0.01-degree box, so this is a windowed read of a COG rather than a full
    tile download, and dst_ellipsoidal_height=False means no geoid grid is
    fetched either -- but it still needs the network, hence 'long'. The
    offline tests below fake stitch_dem out entirely, so this is the only
    place the real call signature is checked against a real installation.
    """
    from dem_stitcher.stitcher import stitch_dem

    zvals, metadata = stitch_dem(
        [-118.26, 34.05, -118.25, 34.06],
        dem_name='glo_30',
        dst_ellipsoidal_height=False,
        dst_area_or_point='Area',
    )

    assert zvals.ndim == 2
    assert zvals.size > 0
    assert metadata['crs'] is not None


# ---------------------------------------------------------------------------
# download_dem internals, with stitch_dem faked out
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_stitcher(monkeypatch):
    """Replace stitch_dem with a synthetic 3x4 DEM and record its arguments.

    Returns a dict that gains 'bounds' and 'kwargs' each time stitch_dem is
    called, so a test can assert both what was requested and -- by the key
    being absent -- that it was never called at all.
    """
    record = {'zvals': np.arange(12, dtype='float32').reshape(3, 4)}
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': 4,
        'height': 3,
        'count': 1,
        'transform': Affine.from_gdal(-118.0, 0.01, 0, 34.0, 0, -0.01),
        'crs': rasterio.crs.CRS.from_epsg(4326),
    }

    def fake_stitch_dem(bounds, **kwargs):
        record['bounds'] = bounds
        record['kwargs'] = kwargs
        return record['zvals'], dict(profile)

    monkeypatch.setattr('RAiDER.dem.stitch_dem', fake_stitch_dem)
    return record


def test_download_dem_snaps_bounds_outward(fake_stitcher, tmp_path):
    """SNWE in, WSEN out, snapped out to whole degrees plus the buffer.

    Worth pinning explicitly: the flooring/ceiling means even a tiny AOI
    requests at least a 1x1 degree tile, which is why download_dem cannot be
    made cheap, and why the live test above calls stitch_dem directly.
    """
    download_dem(ll_bounds=[34.2, 34.8, -118.7, -118.1], dem_path=tmp_path / 'dem.tif', buf=0.02)

    # W = floor(-118.7) - buf, S = floor(34.2) - buf,
    # E = ceil(-118.1) + buf, N = ceil(34.8) + buf
    assert np.allclose(fake_stitcher['bounds'], [-119.02, 33.98, -117.98, 35.02])


def test_download_dem_buffer_is_configurable(fake_stitcher, tmp_path):
    download_dem(ll_bounds=[34.2, 34.8, -118.7, -118.1], dem_path=tmp_path / 'dem.tif', buf=0.5)

    assert np.allclose(fake_stitcher['bounds'], [-119.5, 33.5, -117.5, 35.5])


def test_download_dem_requests_geoid_glo30(fake_stitcher, tmp_path):
    """The datum this whole workflow depends on, pinned as a contract.

    dst_ellipsoidal_height=False leaves GLO-30 on the EGM2008 geoid it is
    distributed against, which is the datum RAiDER reads geoid heights in and
    the datum the weather-model cube samples against -- so the common path
    converts nothing. Flipping this to True would fetch a geoid grid to do a
    conversion that sampling then has to undo.
    """
    download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=tmp_path / 'dem.tif')

    kwargs = fake_stitcher['kwargs']
    assert kwargs['dem_name'] == 'glo_30'
    assert kwargs['dst_ellipsoidal_height'] is False
    assert kwargs['dst_area_or_point'] == 'Area'


def test_download_dem_writes_file_when_requested(fake_stitcher, tmp_path):
    dem_path = tmp_path / 'dem.tif'
    zvals, meta = download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=dem_path, writeDEM=True)

    assert dem_path.exists()
    assert np.allclose(zvals, fake_stitcher['zvals'])
    assert meta is not None

    with rasterio.open(dem_path) as ds:
        assert np.allclose(ds.read(1), fake_stitcher['zvals'])
        # dem.py tags the written file as point-registered
        assert ds.tags()['AREA_OR_POINT'] == 'Point'


def test_download_dem_does_not_write_unless_asked(fake_stitcher, tmp_path):
    dem_path = tmp_path / 'dem.tif'
    download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=dem_path)

    assert not dem_path.exists()


def test_download_dem_reuses_existing_file(fake_stitcher, tmp_path):
    """An existing DEM is read back rather than re-downloaded."""
    dem_path = tmp_path / 'dem.tif'
    download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=dem_path, writeDEM=True)
    assert 'bounds' in fake_stitcher

    del fake_stitcher['bounds']
    zvals, meta = download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=dem_path)

    assert 'bounds' not in fake_stitcher, 'stitch_dem should not be called for an existing DEM'
    assert np.allclose(zvals, fake_stitcher['zvals'])


def test_download_dem_overwrite_forces_redownload(fake_stitcher, tmp_path):
    dem_path = tmp_path / 'dem.tif'
    download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=dem_path, writeDEM=True)

    del fake_stitcher['bounds']
    download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=dem_path, overwrite=True)

    assert 'bounds' in fake_stitcher, 'overwrite=True should re-download over an existing DEM'


def test_download_dem_existing_file_needs_no_bounds(fake_stitcher, tmp_path):
    """ll_bounds is only required when something actually has to be downloaded."""
    dem_path = tmp_path / 'dem.tif'
    download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=dem_path, writeDEM=True)

    zvals, meta = download_dem(dem_path=dem_path)
    assert np.allclose(zvals, fake_stitcher['zvals'])


def test_download_dem_tags_written_file_with_datum(fake_stitcher, tmp_path):
    """The datum tag is what lets a later run tell a cached DEM's vintage."""
    dem_path = tmp_path / 'dem.tif'
    download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=dem_path, writeDEM=True)

    with rasterio.open(dem_path) as ds:
        assert ds.tags()[DEM_DATUM_TAG] == DEM_DATUM


def test_download_dem_warns_on_untagged_cached_dem(fake_stitcher, tmp_path, caplog):
    """A DEM from before this change holds ellipsoidal heights, and nothing in
    the file says so -- reusing it silently would displace every height by the
    geoid undulation, and for StationFile would write that wrong datum into
    the user's CSV as an authoritative label."""
    dem_path = tmp_path / 'dem.tif'
    download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=dem_path, writeDEM=True)

    # Strip the tag to mimic a DEM written by an earlier version
    with rasterio.open(dem_path, 'r+') as ds:
        ds.update_tags(**{DEM_DATUM_TAG: ''})

    with caplog.at_level(logging.WARNING):
        download_dem(dem_path=dem_path)

    assert 'height-datum tag' in caplog.text


def test_download_dem_does_not_warn_on_tagged_cached_dem(fake_stitcher, tmp_path, caplog):
    dem_path = tmp_path / 'dem.tif'
    download_dem(ll_bounds=[34.0, 34.5, -118.5, -118.0], dem_path=dem_path, writeDEM=True)

    with caplog.at_level(logging.WARNING):
        download_dem(dem_path=dem_path)

    assert 'height-datum tag' not in caplog.text
