import logging
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyproj

from test import GEOM_DIR, TEST_DIR
from pyproj import CRS

from RAiDER.cli.raider import calcDelays

from RAiDER.utilFcns import rio_open
from RAiDER.llreader import (
    StationFile, RasterRDR, BoundingBox, GeocodedFile, bounds_from_latlon_rasters,
    bounds_from_csv, _parse_crs, _ellipsoidal_to_geometric,
)

SCENARIO0_DIR = TEST_DIR / "scenario_0"
SCENARIO1_DIR = TEST_DIR / "scenario_1/geom"
SCENARIO2_DIR = TEST_DIR / "scenario_2"


@pytest.fixture
def parser():
    return calcDelays()


@pytest.fixture
def station_file():
    return SCENARIO2_DIR / 'stations.csv'


@pytest.fixture
def llfiles():
    return SCENARIO1_DIR / 'lat.dat', SCENARIO1_DIR / 'lon.dat'


def test_latlon_reader_2():
    with pytest.raises(ValueError):
        RasterRDR(lat_file=None, lon_file=None)

    with pytest.raises(ValueError):
        RasterRDR(lat_file='doesnotexist.rdr', lon_file='doesnotexist.rdr')


def test_aoi_epsg():
    bbox = [20, 27, -115, -104]
    r = BoundingBox(bbox)
    r.set_output_spacing(ll_res=0.05)
    test = r.get_output_spacing(4978)
    assert test == 0.05 * 1e5


def test_set_output_dir():
    bbox = [20, 27, -115, -104]
    r = BoundingBox(bbox)
    r.set_output_directory('dummy_directory')
    assert r._output_directory == 'dummy_directory'


def test_set_xygrid():
    bbox = [20, 27, -115, -104]
    crs = CRS.from_epsg(4326)
    r = BoundingBox(bbox)
    r.set_output_spacing(ll_res=0.1)
    r.set_output_xygrid(dst_crs=4978)
    r.set_output_xygrid(dst_crs=crs)
    assert True


def test_latlon_reader():
    latfile = Path(GEOM_DIR) / 'lat.rdr'
    lonfile = Path(GEOM_DIR) / 'lon.rdr'
    lat_true, _ = rio_open(latfile)
    lon_true, _ = rio_open(lonfile)

    query = RasterRDR(lat_file=str(latfile), lon_file=str(lonfile))
    lats, lons = query.readLL()
    assert lats.shape == (45, 226)
    assert lons.shape == (45, 226)

    assert np.allclose(lat_true, lats, equal_nan=True)
    assert np.allclose(lon_true, lons, equal_nan=True)

    # Hard code the lat/lon bounds to test against changing the files
    bounds_true = [15.7637, 21.4936, -101.6384, -98.2418]
    assert all([np.allclose(b, t, rtol=1e-4) for b, t in zip(query.bounds(), bounds_true)])


def test_badllfiles(station_file):
    latfile = os.path.join(GEOM_DIR, 'lat.rdr')
    lonfile = os.path.join(GEOM_DIR, 'lon_dummy.rdr')
    station_file = station_file
    with pytest.raises(ValueError):
        RasterRDR(lat_file=latfile, lon_file=lonfile)
    with pytest.raises(ValueError):
        RasterRDR(lat_file=latfile, lon_file=station_file)
    with pytest.raises(ValueError):
        RasterRDR(lat_file=station_file, lon_file=lonfile)

def test_read_bbox():
    bbox = [20, 27, -115, -104]
    query = BoundingBox(bbox)
    assert query.type() == 'bounding_box'
    assert query.bounds() == bbox
    assert query.projection() == 'EPSG:4326'


def test_read_station_file(station_file):
    query = StationFile(station_file)
    lats, lons = query.readLL()
    stats = pd.read_csv(station_file).drop_duplicates(subset=["Lat", "Lon"])

    assert np.allclose(lats, stats['Lat'].values)
    assert np.allclose(lons, stats['Lon'].values)

    assert query.projection() == 'EPSG:4326'

    # Hard code the lat/lon bounds to test against changing the files
    bounds_true = [33.746, 36.795, -118.312, -114.892]
    assert all([np.allclose(b, t, rtol=1e-4) for b, t in zip(query.bounds(), bounds_true)])


def test_bounds_from_latlon_rasters():
    lat_path = Path(GEOM_DIR) / 'lat.rdr'
    lon_path = Path(GEOM_DIR) / 'lon.rdr'
    snwe, _, _ = bounds_from_latlon_rasters(str(lat_path), str(lon_path))

    bounds_true =[15.7637, 21.4936, -101.6384, -98.2418]
    assert all([np.allclose(b, t, rtol=1e-4) for b, t in zip(snwe, bounds_true)])


def test_bounds_from_csv(station_file):
    bounds_true = [33.746, 36.795, -118.312, -114.892]
    snwe = bounds_from_csv(station_file)
    assert all([np.allclose(b, t) for b, t in zip(snwe, bounds_true)])


def test_readZ_sf(station_file):
    aoi = StationFile(station_file)
    assert np.allclose(aoi.readZ(), .1)


def test_GeocodedFile():
    aoi = GeocodedFile(SCENARIO0_DIR / 'small_dem.tif', is_dem=True)
    z = aoi.readZ()
    x,y = aoi.readLL()
    assert z.shape == (569,558)
    assert x.shape == z.shape


# ---------------------------------------------------------------------------
# Fixtures shared by CRS / geoid tests
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_station_file(tmp_path):
    """Three-station CSV with known ellipsoidal heights across CONUS latitudes."""
    csv = tmp_path / 'stations.csv'
    csv.write_text(
        'ID,Lat,Lon,Hgt_m\n'
        'STA1,34.0,-118.0,50.0\n'
        'STA2,36.0,-115.0,150.0\n'
        'STA3,49.0,-122.0,195.28\n'
    )
    return csv


@pytest.fixture
def conus_lats():
    return np.array([34.0, 36.0, 49.0])


@pytest.fixture
def conus_lons():
    return np.array([-118.0, -115.0, -122.0])


@pytest.fixture
def conus_heights():
    return np.array([50.0, 150.0, 195.28])


# ---------------------------------------------------------------------------
# _parse_crs
# ---------------------------------------------------------------------------

def test_parse_crs_int():
    assert _parse_crs(4326).to_epsg() == 4326


def test_parse_crs_string_bare():
    assert _parse_crs('4979').to_epsg() == 4979


def test_parse_crs_string_with_prefix():
    assert _parse_crs('EPSG:4979').to_epsg() == 4979


def test_parse_crs_crs_object_is_identity():
    """Passing a CRS object should return the same object unchanged."""
    src = CRS.from_epsg(32631)
    assert _parse_crs(src) is src


def test_parse_crs_invalid_raises():
    with pytest.raises(Exception):
        _parse_crs('definitely_not_a_crs')


# ---------------------------------------------------------------------------
# _ellipsoidal_to_geometric – unit tests (no network required)
# ---------------------------------------------------------------------------

def test_ellipsoidal_to_geometric_2d_crs_is_noop(conus_lats, conus_lons, conus_heights):
    """A 2D CRS (EPSG:4326) must return heights unchanged without touching PROJ."""
    result = _ellipsoidal_to_geometric(
        conus_lats, conus_lons, conus_heights, CRS.from_epsg(4326)
    )
    assert np.array_equal(result, conus_heights)


def test_ellipsoidal_to_geometric_cdn_fallback_applied(
    conus_lats, conus_lons, conus_heights, monkeypatch
):
    """Noop local grid → CDN enabled → real corrections applied."""
    correction = np.full_like(conus_heights, 20.0)
    h_expected = conus_heights + correction
    call_count = {'n': 0}

    def mock_from_crs(*args, **kwargs):
        call_count['n'] += 1
        t = MagicMock()
        if call_count['n'] == 1:       # first build: local grid missing → noop
            t.to_proj4.return_value = '+proj=noop'
            t.transform.return_value = (conus_lons, conus_lats, conus_heights.copy())
        else:                           # second build: after CDN enabled → real grid
            t.to_proj4.return_value = None
            t.transform.return_value = (conus_lons, conus_lats, h_expected.copy())
        return t

    enabled_states = []
    monkeypatch.setattr(pyproj.Transformer, 'from_crs', mock_from_crs)
    monkeypatch.setattr(pyproj.network, 'is_network_enabled', lambda: False)
    monkeypatch.setattr(pyproj.network, 'set_network_enabled',
                        lambda v: enabled_states.append(v))

    result = _ellipsoidal_to_geometric(
        conus_lats, conus_lons, conus_heights, CRS.from_epsg(4979)
    )

    assert np.allclose(result, h_expected)
    assert call_count['n'] == 2, 'Transformer should be built twice (local then CDN)'
    assert True in enabled_states, 'CDN should have been enabled'
    assert False in enabled_states, 'CDN state should have been restored'


def test_ellipsoidal_to_geometric_cdn_also_noop_warns(
    conus_lats, conus_lons, conus_heights, monkeypatch, caplog
):
    """When CDN is also unavailable (still noop), return heights with a warning."""
    noop_mock = MagicMock()
    noop_mock.to_proj4.return_value = '+proj=noop'
    noop_mock.transform.return_value = (conus_lons, conus_lats, conus_heights.copy())

    monkeypatch.setattr(pyproj.Transformer, 'from_crs', lambda *a, **kw: noop_mock)
    monkeypatch.setattr(pyproj.network, 'is_network_enabled', lambda: False)
    monkeypatch.setattr(pyproj.network, 'set_network_enabled', lambda v: None)

    with caplog.at_level(logging.WARNING, logger='RAiDER'):
        result = _ellipsoidal_to_geometric(
            conus_lats, conus_lons, conus_heights, CRS.from_epsg(4979)
        )

    assert np.array_equal(result, conus_heights)
    assert 'EGM96 grid unavailable' in caplog.text


def test_ellipsoidal_to_geometric_exception_returns_unchanged(
    conus_lats, conus_lons, conus_heights, monkeypatch, caplog
):
    """An unexpected PROJ exception must return heights unchanged with a warning."""
    def raising(*args, **kwargs):
        raise RuntimeError('simulated PROJ error')

    monkeypatch.setattr(pyproj.Transformer, 'from_crs', raising)

    with caplog.at_level(logging.WARNING, logger='RAiDER'):
        result = _ellipsoidal_to_geometric(
            conus_lats, conus_lons, conus_heights, CRS.from_epsg(4979)
        )

    assert np.array_equal(result, conus_heights)
    assert 'conversion failed' in caplog.text.lower()


# ---------------------------------------------------------------------------
# _ellipsoidal_to_geometric – integration test (requires PROJ CDN or local grids)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_ellipsoidal_to_geometric_egm96_conus_values(
    conus_lats, conus_lons, conus_heights
):
    """EGM96 orthometric heights in CONUS are 10–45 m larger than WGS84 ellipsoidal."""
    result = _ellipsoidal_to_geometric(
        conus_lats, conus_lons, conus_heights, CRS.from_epsg(4979)
    )
    corrections = result - conus_heights
    # EGM96 geoid is 15–35 m below the WGS84 ellipsoid in CONUS, so
    # orthometric heights are larger than ellipsoidal heights.
    assert np.all(corrections > 10), f'Expected >10 m corrections; got {corrections}'
    assert np.all(corrections < 45), f'Expected <45 m corrections; got {corrections}'


# ---------------------------------------------------------------------------
# StationFile – crs parameter and update_crs method
# ---------------------------------------------------------------------------

def test_station_file_default_crs_is_4326(tmp_station_file):
    sf = StationFile(tmp_station_file)
    assert sf._crs.to_epsg() == 4326


def test_station_file_crs_int(tmp_station_file):
    sf = StationFile(tmp_station_file, crs=4979)
    assert sf._crs.to_epsg() == 4979


def test_station_file_crs_string(tmp_station_file):
    sf = StationFile(tmp_station_file, crs='4979')
    assert sf._crs.to_epsg() == 4979


def test_station_file_crs_epsg_string(tmp_station_file):
    sf = StationFile(tmp_station_file, crs='EPSG:4979')
    assert sf._crs.to_epsg() == 4979


def test_station_file_crs_object(tmp_station_file):
    crs = CRS.from_epsg(4979)
    sf = StationFile(tmp_station_file, crs=crs)
    assert sf._crs.to_epsg() == 4979


def test_update_crs_changes_datum(tmp_station_file):
    sf = StationFile(tmp_station_file)
    assert sf._crs.to_epsg() == 4326
    sf.update_crs(4979)
    assert sf._crs.to_epsg() == 4979


def test_update_crs_string_input(tmp_station_file):
    sf = StationFile(tmp_station_file)
    sf.update_crs('EPSG:4979')
    assert sf._crs.to_epsg() == 4979


def test_update_crs_crs_object(tmp_station_file):
    sf = StationFile(tmp_station_file)
    sf.update_crs(CRS.from_epsg(4979))
    assert sf._crs.to_epsg() == 4979


# ---------------------------------------------------------------------------
# StationFile.readZ – height conversion plumbing
# ---------------------------------------------------------------------------

def test_readZ_default_crs_heights_unchanged(tmp_station_file):
    """crs=4326 must return Hgt_m values from the CSV unmodified."""
    z = StationFile(tmp_station_file).readZ()
    assert np.allclose(z, [50.0, 150.0, 195.28])


def test_readZ_3d_crs_calls_conversion(tmp_station_file, monkeypatch):
    """crs=4979 must route Hgt_m through _ellipsoidal_to_geometric."""
    import RAiDER.llreader as lr

    captured = {}

    def mock_convert(lats, lons, heights, crs):
        captured['heights'] = heights.copy()
        captured['crs_epsg'] = crs.to_epsg()
        return heights + 25.0

    monkeypatch.setattr(lr, '_ellipsoidal_to_geometric', mock_convert)

    z = StationFile(tmp_station_file, crs=4979).readZ()

    assert captured['crs_epsg'] == 4979
    assert np.allclose(captured['heights'], [50.0, 150.0, 195.28])
    assert np.allclose(z, [75.0, 175.0, 220.28])


def test_readZ_update_crs_uses_new_datum(tmp_station_file, monkeypatch):
    """update_crs before readZ must pass the updated CRS to the conversion."""
    import RAiDER.llreader as lr

    captured_epsg = []

    def mock_convert(lats, lons, heights, crs):
        captured_epsg.append(crs.to_epsg())
        return heights

    monkeypatch.setattr(lr, '_ellipsoidal_to_geometric', mock_convert)

    sf = StationFile(tmp_station_file)   # default 4326
    sf.update_crs(4979)
    sf.readZ()

    assert captured_epsg == [4979]


def test_readZ_2d_crs_conversion_is_noop(tmp_station_file, monkeypatch):
    """_ellipsoidal_to_geometric is still called for 2D CRS but returns heights unchanged."""
    import RAiDER.llreader as lr

    call_count = {'n': 0}

    def mock_convert(lats, lons, heights, crs):
        call_count['n'] += 1
        # The function itself handles the 2D noop, but verify it receives crs=4326
        assert crs.to_epsg() == 4326
        return heights  # same as real behaviour for 2D

    monkeypatch.setattr(lr, '_ellipsoidal_to_geometric', mock_convert)

    z = StationFile(tmp_station_file).readZ()

    assert call_count['n'] == 1
    assert np.allclose(z, [50.0, 150.0, 195.28])
