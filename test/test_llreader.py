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
    _parse_crs, _ellipsoidal_to_geometric, _geometric_to_ellipsoidal, _source_is_geoid,
    HGT_DATUM_COLUMN,
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
    assert query._crs == CRS.from_epsg(4979)


def test_stationfile_geoid_crs(station_file):
    query = StationFile(station_file, crs=4326)
    assert query._crs == CRS.from_epsg(4326)


def test_stationfile_update_crs(station_file):
    query = StationFile(station_file)
    query.update_crs(4326)
    assert query._crs == CRS.from_epsg(4326)

    query.update_crs(4979)
    assert query._crs == CRS.from_epsg(4979)


def test_readZ_sf_default_ellipsoidal_no_conversion(station_file):
    """crs=4979 (ellipsoidal, the default): readZ()'s default request needs no conversion."""
    query = StationFile(station_file, crs=4979)
    assert np.allclose(query.readZ(), 0.1)


def test_readZ_sf_default_converts_from_declared_geoid(monkeypatch, station_file):
    """crs=4326 (geoid): readZ()'s default (ellipsoidal) request routes through _geometric_to_ellipsoidal."""
    calls = {}

    def fake_convert(lats, lons, heights):
        calls['called'] = True
        return heights + 5.0

    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fake_convert)

    query = StationFile(station_file, crs=4326)
    z = query.readZ()

    assert calls.get('called') is True
    assert np.allclose(z, 0.1 + 5.0)


def test_readZ_sf_geoid_request_no_conversion_when_declared_geoid(station_file):
    """crs=4326 (geoid): readZ(geoid_heights=True) needs no conversion."""
    query = StationFile(station_file, crs=4326)
    assert np.allclose(query.readZ(geoid_heights=True), 0.1)


def test_readZ_sf_geoid_request_converts_from_ellipsoidal(monkeypatch, station_file):
    """crs=4979 (ellipsoidal, the default): readZ(geoid_heights=True) routes through _ellipsoidal_to_geometric."""
    calls = {}

    def fake_convert(lats, lons, heights, crs):
        calls['called'] = True
        calls['crs'] = crs
        return heights + 5.0

    monkeypatch.setattr('RAiDER.llreader._ellipsoidal_to_geometric', fake_convert)

    query = StationFile(station_file, crs=4979)
    z = query.readZ(geoid_heights=True)

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
# StationFile.readZ -- no-Hgt_m / DEM-download branch
# ---------------------------------------------------------------------------

def test_readZ_sf_dem_download_default_ellipsoidal_no_conversion(monkeypatch, tmp_path):
    """DEM-derived heights are ellipsoidal at the source; the default request needs no conversion."""
    csv_path = tmp_path / 'stations_no_hgt.csv'
    csv_path.write_text('ID,Lat,Lon\nA,34.0,-118.0\nB,35.0,-117.0\n')

    def fake_download_dem(*args, **kwargs):
        return None, None

    def fake_interpolateDEM(dem_file, ll):
        lats, _ = ll
        n = len(lats)
        return np.eye(n) * 100.0  # diagonal = station heights

    monkeypatch.setattr('RAiDER.dem.download_dem', fake_download_dem)
    monkeypatch.setattr('RAiDER.interpolator.interpolateDEM', fake_interpolateDEM)

    query = StationFile(csv_path)
    z = query.readZ()

    assert np.allclose(z, 100.0)
    # self._crs must be updated to reflect the DEM's actual (ellipsoidal) datum,
    # so a later readZ() call on this same object doesn't misread it -- this is
    # the exact state trap that caused the stale-_crs bug.
    assert query._crs == CRS.from_epsg(4979)


def test_readZ_sf_dem_download_geoid_request_converts(monkeypatch, tmp_path):
    """Requesting geoid_heights=True on the DEM branch converts via _ellipsoidal_to_geometric."""
    csv_path = tmp_path / 'stations_no_hgt.csv'
    csv_path.write_text('ID,Lat,Lon\nA,34.0,-118.0\nB,35.0,-117.0\n')

    def fake_download_dem(*args, **kwargs):
        return None, None

    def fake_interpolateDEM(dem_file, ll):
        lats, _ = ll
        n = len(lats)
        return np.eye(n) * 100.0  # diagonal = station heights

    calls = {}

    def fake_convert(lats, lons, heights, crs):
        calls['heights'] = heights
        calls['crs'] = crs
        return heights + 7.0

    monkeypatch.setattr('RAiDER.dem.download_dem', fake_download_dem)
    monkeypatch.setattr('RAiDER.interpolator.interpolateDEM', fake_interpolateDEM)
    monkeypatch.setattr('RAiDER.llreader._ellipsoidal_to_geometric', fake_convert)

    query = StationFile(csv_path)
    z = query.readZ(geoid_heights=True)

    assert np.allclose(calls['heights'], 100.0)
    assert calls['crs'] == CRS.from_epsg(4979)
    assert np.allclose(z, 100.0 + 7.0)


def test_readZ_sf_write_back_then_read_again_stays_consistent(tmp_path, monkeypatch):
    """Regression test for the stale-_crs bug: write-back via the DEM branch, then
    read again (both directions) on the same object, without going through PROJ at
    all -- crs=4979 declared up front matches what the DEM branch would set anyway,
    so both reads should agree with the DEM heights with no double-conversion.
    """
    csv_path = tmp_path / 'stations_no_hgt.csv'
    csv_path.write_text('ID,Lat,Lon\nA,34.0,-118.0\nB,35.0,-117.0\n')

    def fake_download_dem(*args, **kwargs):
        return None, None

    def fake_interpolateDEM(dem_file, ll):
        lats, _ = ll
        return np.eye(len(lats)) * 100.0  # diagonal = station heights

    monkeypatch.setattr('RAiDER.dem.download_dem', fake_download_dem)
    monkeypatch.setattr('RAiDER.interpolator.interpolateDEM', fake_interpolateDEM)

    query = StationFile(csv_path, crs=4979, output_directory=tmp_path)

    z1 = query.readZ()  # triggers the DEM-download branch, writes ellipsoidal Hgt_m
    z2 = query.readZ()  # Hgt_m now exists; must still read as ellipsoidal, not geoid

    assert np.allclose(z1, z2)
    assert query._crs == CRS.from_epsg(4979)


# ---------------------------------------------------------------------------
# RasterRDR.readZ
# ---------------------------------------------------------------------------

def test_rasterrdr_readZ_default_converts_from_hgtfile_geoid(monkeypatch):
    """hgt_file is assumed geoid; the default (ellipsoidal) request converts via _geometric_to_ellipsoidal."""
    latfile = Path(GEOM_DIR) / 'lat.rdr'
    lonfile = Path(GEOM_DIR) / 'lon.rdr'
    query = RasterRDR(lat_file=str(latfile), lon_file=str(lonfile), hgt_file=str(latfile))

    calls = {}

    def fake_convert(lats, lons, heights):
        calls['called'] = True
        calls['heights'] = heights
        return heights + 3.0

    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fake_convert)

    z = query.readZ()

    assert calls.get('called') is True
    assert np.allclose(z, calls['heights'] + 3.0)


def test_rasterrdr_readZ_geoid_request_no_conversion(monkeypatch):
    """hgt_file is assumed geoid; requesting geoid_heights=True needs no conversion."""
    latfile = Path(GEOM_DIR) / 'lat.rdr'
    lonfile = Path(GEOM_DIR) / 'lon.rdr'
    query = RasterRDR(lat_file=str(latfile), lon_file=str(lonfile), hgt_file=str(latfile))
    raw_hgts, _ = rio_open(latfile)

    def fail_if_called(*args, **kwargs):
        raise AssertionError('conversion should not be called')

    monkeypatch.setattr('RAiDER.llreader._ellipsoidal_to_geometric', fail_if_called)
    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fail_if_called)

    z = query.readZ(geoid_heights=True)
    assert np.allclose(z, raw_hgts, equal_nan=True)


def test_rasterrdr_readZ_default_requires_lonfile():
    """Without a separate lon_file, readLL() can't provide lons for the conversion
    the default (ellipsoidal) request now needs, since hgt_file is assumed geoid."""
    latfile = Path(GEOM_DIR) / 'lat.rdr'
    lonfile = Path(GEOM_DIR) / 'lon.rdr'
    query = RasterRDR(lat_file=str(latfile), lon_file=str(lonfile), hgt_file=str(latfile))
    query._lonfile = None

    with pytest.raises(NotImplementedError):
        query.readZ()


# ---------------------------------------------------------------------------
# GeocodedFile.readZ
# ---------------------------------------------------------------------------

def test_GeocodedFile_readZ_default_converts_from_existing_dem_geoid(monkeypatch):
    """is_dem=True with an existing file is assumed geoid; the default (ellipsoidal)
    request converts via _geometric_to_ellipsoidal."""
    aoi = GeocodedFile(SCENARIO0_DIR / 'small_dem.tif', is_dem=True)

    calls = {}

    def fake_convert(lats, lons, heights):
        calls['called'] = True
        return heights + 2.0

    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fake_convert)

    z_geoid = aoi.readZ(geoid_heights=True)
    z_ell = aoi.readZ()

    assert calls.get('called') is True
    # small_dem.tif has no geotransform, so interpolateDEM returns all-NaN heights here;
    # equal_nan=True since we're only checking the mocked offset was applied elementwise.
    assert np.allclose(z_ell, z_geoid + 2.0, equal_nan=True)


# ---------------------------------------------------------------------------
# _source_is_geoid / Hgt_datum column
# ---------------------------------------------------------------------------

def test_source_is_geoid_falls_back_to_crs_when_no_column():
    df = pd.DataFrame({'Lat': [34.0], 'Lon': [-118.0], 'Hgt_m': [100.0]})
    assert _source_is_geoid(df, CRS.from_epsg(4326)) is True
    assert _source_is_geoid(df, CRS.from_epsg(4979)) is False


def test_source_is_geoid_column_overrides_crs():
    """The column is authoritative even when it contradicts the passed-in crs."""
    df = pd.DataFrame({'Lat': [34.0], 'Lon': [-118.0], 'Hgt_m': [100.0], HGT_DATUM_COLUMN: ['geoid']})
    assert _source_is_geoid(df, CRS.from_epsg(4979)) is True

    df[HGT_DATUM_COLUMN] = 'ellipsoidal'
    assert _source_is_geoid(df, CRS.from_epsg(4326)) is False


def test_source_is_geoid_inconsistent_column_raises():
    df = pd.DataFrame({
        'Lat': [34.0, 35.0], 'Lon': [-118.0, -117.0], 'Hgt_m': [100.0, 200.0],
        HGT_DATUM_COLUMN: ['ellipsoidal', 'geoid'],
    })
    with pytest.raises(ValueError):
        _source_is_geoid(df, CRS.from_epsg(4979))


def test_source_is_geoid_unrecognized_value_raises():
    df = pd.DataFrame({'Lat': [34.0], 'Lon': [-118.0], 'Hgt_m': [100.0], HGT_DATUM_COLUMN: ['orthometric']})
    with pytest.raises(ValueError):
        _source_is_geoid(df, CRS.from_epsg(4979))


def test_readZ_sf_dem_download_writes_datum_column(tmp_path, monkeypatch):
    csv_path = tmp_path / 'stations_no_hgt.csv'
    csv_path.write_text('ID,Lat,Lon\nA,34.0,-118.0\nB,35.0,-117.0\n')

    def fake_download_dem(*args, **kwargs):
        return None, None

    def fake_interpolateDEM(dem_file, ll):
        lats, _ = ll
        return np.eye(len(lats)) * 100.0

    monkeypatch.setattr('RAiDER.dem.download_dem', fake_download_dem)
    monkeypatch.setattr('RAiDER.interpolator.interpolateDEM', fake_interpolateDEM)

    StationFile(csv_path).readZ()

    df = pd.read_csv(csv_path)
    assert HGT_DATUM_COLUMN in df.columns
    assert (df[HGT_DATUM_COLUMN] == 'ellipsoidal').all()


def test_readZ_sf_datum_column_protects_against_wrong_crs(tmp_path, monkeypatch):
    """A second StationFile pointed at the same file, constructed with the wrong
    crs=, should still read correctly because the Hgt_datum column overrides it --
    this is the exact scenario item 1's CSV marker exists to prevent.
    """
    csv_path = tmp_path / 'stations_no_hgt.csv'
    csv_path.write_text('ID,Lat,Lon\nA,34.0,-118.0\nB,35.0,-117.0\n')

    def fake_download_dem(*args, **kwargs):
        return None, None

    def fake_interpolateDEM(dem_file, ll):
        lats, _ = ll
        return np.eye(len(lats)) * 100.0

    monkeypatch.setattr('RAiDER.dem.download_dem', fake_download_dem)
    monkeypatch.setattr('RAiDER.interpolator.interpolateDEM', fake_interpolateDEM)

    # First object downloads the DEM (ellipsoidal) and writes it, with the datum column.
    StationFile(csv_path).readZ()

    # A second, independent object is constructed with a *wrong* crs -- geoid,
    # when the file actually holds ellipsoidal heights. Without the column this
    # would silently misread the file; with it, the column wins.
    second = StationFile(csv_path, crs=4326)
    assert np.allclose(second.readZ(), 100.0)  # ellipsoidal (default) request, no conversion needed


# ---------------------------------------------------------------------------
# Loud warning on missing-grid / conversion failure
# ---------------------------------------------------------------------------

def test_convert_missing_grid_raises_user_warning(monkeypatch):
    class NoopTransformer:
        def to_proj4(self):
            return '+proj=noop'

        def transform(self, lons, lats, heights):
            return lons, lats, heights

    monkeypatch.setattr(pyproj.Transformer, 'from_crs', lambda src, dst, always_xy=True: NoopTransformer())
    monkeypatch.setattr(pyproj.network, 'is_network_enabled', lambda: False)
    monkeypatch.setattr(pyproj.network, 'set_network_enabled', lambda v: None)

    heights = np.array([100.0])
    with pytest.warns(UserWarning, match='Height datum conversion unavailable'):
        result = _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), heights)

    assert np.array_equal(result, heights)


def test_convert_exception_raises_user_warning(monkeypatch):
    def raise_err(src, dst, always_xy=True):
        raise RuntimeError('boom')

    monkeypatch.setattr(pyproj.Transformer, 'from_crs', raise_err)

    heights = np.array([100.0])
    with pytest.warns(UserWarning, match='Height datum conversion unavailable'):
        result = _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), heights)

    assert np.array_equal(result, heights)
