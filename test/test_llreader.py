import os
from pathlib import Path
import pytest

import numpy as np
import pandas as pd
import pyproj

from test import GEOM_DIR, TEST_DIR
from pyproj import CRS

from RAiDER.cli.raider import calcDelays

from RAiDER.utilFcns import rio_open
from RAiDER.llreader import (
    StationFile, RasterRDR, BoundingBox, GeocodedFile, bounds_from_latlon_rasters, bounds_from_csv,
    _parse_crs, _ellipsoidal_to_geometric, _geometric_to_ellipsoidal,
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

    bounds_true = [15.7637, 21.4936, -101.6384, -98.2418]
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
    x, y = aoi.readLL()
    assert z.shape == (569, 558)
    assert x.shape == z.shape


# ---------------------------------------------------------------------------
# _parse_crs
# ---------------------------------------------------------------------------

def test_parse_crs_passes_through_crs_object():
    crs_obj = CRS.from_epsg(4326)
    assert _parse_crs(crs_obj) is crs_obj


def test_parse_crs_from_int_epsg():
    assert _parse_crs(4979) == CRS.from_epsg(4979)


def test_parse_crs_from_numeric_string():
    assert _parse_crs('4326') == CRS.from_epsg(4326)


def test_parse_crs_from_crs_string():
    # "EPSG:4326" is not a bare EPSG code, so from_epsg() fails and the
    # function should fall back to the general CRS() constructor.
    assert _parse_crs('EPSG:4326') == CRS.from_epsg(4326)


def test_parse_crs_invalid_string_raises():
    with pytest.raises(pyproj.exceptions.CRSError):
        _parse_crs('not_a_real_crs')


# ---------------------------------------------------------------------------
# _ellipsoidal_to_geometric
# ---------------------------------------------------------------------------

def test_ellipsoidal_to_geometric_2d_crs_passthrough():
    """A 2D CRS (e.g. EPSG:4326) carries no vertical datum, so heights are unchanged."""
    lats = np.array([34.0, 35.0])
    lons = np.array([-118.0, -117.0])
    heights = np.array([100.0, 200.0])

    result = _ellipsoidal_to_geometric(lats, lons, heights, CRS.from_epsg(4326))

    assert result is heights
    assert np.array_equal(result, heights)


def test_ellipsoidal_to_geometric_successful_conversion(monkeypatch):
    """When PROJ has the EGM96 grid, heights should be converted on the first try."""
    class FakeTransformer:
        def to_proj4(self):
            return '+proj=pipeline'  # anything other than '+proj=noop'

        def transform(self, lons, lats, heights):
            return lons, lats, heights + 20.0

    network_calls = []
    monkeypatch.setattr(pyproj.Transformer, 'from_crs', lambda crs, target, always_xy=True: FakeTransformer())
    monkeypatch.setattr(pyproj.network, 'set_network_enabled', lambda v: network_calls.append(v))

    lats = np.array([34.0, 35.0])
    lons = np.array([-118.0, -117.0])
    heights = np.array([100.0, 200.0])

    result = _ellipsoidal_to_geometric(lats, lons, heights, CRS.from_epsg(4979))

    assert np.allclose(result, heights + 20.0)
    # Network was never touched since the first attempt already succeeded.
    assert network_calls == []


def test_ellipsoidal_to_geometric_missing_grid_falls_back(monkeypatch):
    """If the EGM96 grid is missing locally and the CDN is unreachable, heights pass through."""
    class NoopTransformer:
        def to_proj4(self):
            return '+proj=noop'

        def transform(self, lons, lats, heights):
            return lons, lats, heights

    network_state = {'enabled': False}
    set_calls = []

    def fake_set_network_enabled(v):
        set_calls.append(v)
        network_state['enabled'] = v

    monkeypatch.setattr(pyproj.Transformer, 'from_crs', lambda crs, target, always_xy=True: NoopTransformer())
    monkeypatch.setattr(pyproj.network, 'is_network_enabled', lambda: network_state['enabled'])
    monkeypatch.setattr(pyproj.network, 'set_network_enabled', fake_set_network_enabled)

    lats = np.array([34.0])
    lons = np.array([-118.0])
    heights = np.array([100.0])

    result = _ellipsoidal_to_geometric(lats, lons, heights, CRS.from_epsg(4979))

    assert np.array_equal(result, heights)
    # Network enabled to attempt the CDN download, then restored to its prior state.
    assert set_calls == [True, False]


def test_ellipsoidal_to_geometric_exception_falls_back(monkeypatch):
    """Any unexpected failure during the transform should not crash readZ."""
    def raise_err(crs, target, always_xy=True):
        raise RuntimeError('boom')

    monkeypatch.setattr(pyproj.Transformer, 'from_crs', raise_err)

    lats = np.array([34.0])
    lons = np.array([-118.0])
    heights = np.array([100.0])

    result = _ellipsoidal_to_geometric(lats, lons, heights, CRS.from_epsg(4979))

    assert np.array_equal(result, heights)


# ---------------------------------------------------------------------------
# StationFile crs handling
# ---------------------------------------------------------------------------

def test_stationfile_default_crs(station_file):
    query = StationFile(station_file)
    assert query._crs == CRS.from_epsg(4326)


def test_stationfile_ellipsoidal_crs(station_file):
    query = StationFile(station_file, crs=4979)
    assert query._crs == CRS.from_epsg(4979)


def test_stationfile_update_crs(station_file):
    query = StationFile(station_file)
    query.update_crs(4979)
    assert query._crs == CRS.from_epsg(4979)

    query.update_crs(4326)
    assert query._crs == CRS.from_epsg(4326)


def test_readZ_sf_converts_when_ellipsoidal(monkeypatch, station_file):
    """readZ should route through _ellipsoidal_to_geometric when crs=4979."""
    calls = {}

    def fake_convert(lats, lons, heights, crs):
        calls['called'] = True
        calls['crs'] = crs
        return heights + 5.0

    monkeypatch.setattr('RAiDER.llreader._ellipsoidal_to_geometric', fake_convert)

    query = StationFile(station_file, crs=4979)
    z = query.readZ()

    assert calls.get('called') is True
    assert calls['crs'] == CRS.from_epsg(4979)
    assert np.allclose(z, 0.1 + 5.0)


# ---------------------------------------------------------------------------
# _geometric_to_ellipsoidal
# ---------------------------------------------------------------------------

def test_geometric_to_ellipsoidal_successful_conversion(monkeypatch):
    """When PROJ has the EGM96 grid, heights should be converted on the first try."""
    class FakeTransformer:
        def to_proj4(self):
            return '+proj=pipeline'  # anything other than '+proj=noop'

        def transform(self, lons, lats, heights):
            return lons, lats, heights - 15.0

    captured = {}

    def fake_from_crs(src_crs, dst_crs, always_xy=True):
        captured['src'] = src_crs
        captured['dst'] = dst_crs
        return FakeTransformer()

    monkeypatch.setattr(pyproj.Transformer, 'from_crs', fake_from_crs)

    lats = np.array([34.0, 35.0])
    lons = np.array([-118.0, -117.0])
    heights = np.array([100.0, 200.0])  # geoid heights

    result = _geometric_to_ellipsoidal(lats, lons, heights)

    assert np.allclose(result, heights - 15.0)
    # Default target is WGS84 ellipsoidal, converting from the EGM96 geoid.
    assert captured['src'] == CRS.from_epsg(9707)
    assert captured['dst'] == CRS.from_epsg(4979)


def test_geometric_to_ellipsoidal_custom_target_crs(monkeypatch):
    """A caller-supplied target CRS should be used instead of the default."""
    class FakeTransformer:
        def to_proj4(self):
            return '+proj=pipeline'

        def transform(self, lons, lats, heights):
            return lons, lats, heights

    captured = {}

    def fake_from_crs(src_crs, dst_crs, always_xy=True):
        captured['dst'] = dst_crs
        return FakeTransformer()

    monkeypatch.setattr(pyproj.Transformer, 'from_crs', fake_from_crs)

    custom_crs = CRS.from_epsg(4979)
    _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), np.array([100.0]), crs=custom_crs)

    assert captured['dst'] == custom_crs


def test_geometric_to_ellipsoidal_missing_grid_falls_back(monkeypatch):
    """If the EGM96 grid is missing locally and the CDN is unreachable, heights pass through."""
    class NoopTransformer:
        def to_proj4(self):
            return '+proj=noop'

        def transform(self, lons, lats, heights):
            return lons, lats, heights

    network_state = {'enabled': False}
    set_calls = []

    def fake_set_network_enabled(v):
        set_calls.append(v)
        network_state['enabled'] = v

    monkeypatch.setattr(pyproj.Transformer, 'from_crs', lambda src, dst, always_xy=True: NoopTransformer())
    monkeypatch.setattr(pyproj.network, 'is_network_enabled', lambda: network_state['enabled'])
    monkeypatch.setattr(pyproj.network, 'set_network_enabled', fake_set_network_enabled)

    heights = np.array([100.0])
    result = _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), heights)

    assert np.array_equal(result, heights)
    assert set_calls == [True, False]


def test_geometric_to_ellipsoidal_exception_falls_back(monkeypatch):
    """Any unexpected failure during the transform should not crash readZ."""
    def raise_err(src, dst, always_xy=True):
        raise RuntimeError('boom')

    monkeypatch.setattr(pyproj.Transformer, 'from_crs', raise_err)

    heights = np.array([100.0])
    result = _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), heights)

    assert np.array_equal(result, heights)


# ---------------------------------------------------------------------------
# StationFile.readZ(ellipsoidal_heights=True)
# ---------------------------------------------------------------------------

def test_readZ_sf_ellipsoidal_when_crs_already_ellipsoidal(station_file):
    """If self._crs is already 3D-ellipsoidal, raw Hgt_m values should pass through unconverted."""
    query = StationFile(station_file, crs=4979)
    z = query.readZ(ellipsoidal_heights=True)
    assert np.allclose(z, 0.1)


def test_readZ_sf_ellipsoidal_converts_from_geoid(monkeypatch, station_file):
    """If self._crs is geoid (default), ellipsoidal_heights=True should convert via _geometric_to_ellipsoidal."""
    calls = {}

    def fake_convert(lats, lons, heights):
        calls['called'] = True
        calls['heights'] = heights
        return heights + 10.0

    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fake_convert)

    query = StationFile(station_file)  # default crs=4326 (geoid)
    z = query.readZ(ellipsoidal_heights=True)

    assert calls.get('called') is True
    assert np.allclose(z, 0.1 + 10.0)


def test_readZ_sf_dem_download_ellipsoidal(monkeypatch, tmp_path):
    """The no-Hgt_m/DEM-download branch should also convert to ellipsoidal on request."""
    csv_path = tmp_path / 'stations_no_hgt.csv'
    csv_path.write_text('ID,Lat,Lon\nA,34.0,-118.0\nB,35.0,-117.0\n')

    def fake_download_dem(*args, **kwargs):
        return None, None

    def fake_interpolateDEM(dem_file, ll):
        lats, _ = ll
        n = len(lats)
        return np.eye(n) * 100.0  # diagonal = station heights

    calls = {}

    def fake_convert(lats, lons, heights):
        calls['heights'] = heights
        return heights + 7.0

    monkeypatch.setattr('RAiDER.dem.download_dem', fake_download_dem)
    monkeypatch.setattr('RAiDER.interpolator.interpolateDEM', fake_interpolateDEM)
    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fake_convert)

    query = StationFile(csv_path)
    z = query.readZ(ellipsoidal_heights=True)

    assert np.allclose(calls['heights'], 100.0)
    assert np.allclose(z, 100.0 + 7.0)


# ---------------------------------------------------------------------------
# RasterRDR.readZ(ellipsoidal_heights=True)
# ---------------------------------------------------------------------------

def test_rasterrdr_readZ_ellipsoidal_converts(monkeypatch):
    latfile = Path(GEOM_DIR) / 'lat.rdr'
    lonfile = Path(GEOM_DIR) / 'lon.rdr'
    query = RasterRDR(lat_file=str(latfile), lon_file=str(lonfile), hgt_file=str(latfile))

    calls = {}

    def fake_convert(lats, lons, heights):
        calls['called'] = True
        calls['heights'] = heights
        return heights + 3.0

    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fake_convert)

    z = query.readZ(ellipsoidal_heights=True)

    assert calls.get('called') is True
    assert np.allclose(z, calls['heights'] + 3.0)


def test_rasterrdr_readZ_ellipsoidal_requires_lonfile():
    """Without a separate lon_file, readLL() can't provide lons for the conversion."""
    latfile = Path(GEOM_DIR) / 'lat.rdr'
    lonfile = Path(GEOM_DIR) / 'lon.rdr'
    query = RasterRDR(lat_file=str(latfile), lon_file=str(lonfile), hgt_file=str(latfile))
    query._lonfile = None

    with pytest.raises(NotImplementedError):
        query.readZ(ellipsoidal_heights=True)


# ---------------------------------------------------------------------------
# GeocodedFile.readZ(ellipsoidal_heights=True)
# ---------------------------------------------------------------------------

def test_GeocodedFile_readZ_ellipsoidal_converts(monkeypatch):
    aoi = GeocodedFile(SCENARIO0_DIR / 'small_dem.tif', is_dem=True)

    calls = {}

    def fake_convert(lats, lons, heights):
        calls['called'] = True
        return heights + 2.0

    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fake_convert)

    z_geoid = aoi.readZ()
    z_ell = aoi.readZ(ellipsoidal_heights=True)

    assert calls.get('called') is True
    # small_dem.tif has no geotransform, so interpolateDEM returns all-NaN heights here;
    # equal_nan=True since we're only checking the mocked offset was applied elementwise.
    assert np.allclose(z_ell, z_geoid + 2.0, equal_nan=True)
