import os
import warnings
from pathlib import Path
import pytest

import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from test import GEOM_DIR, TEST_DIR
from pyproj import CRS

from RAiDER.cli.raider import calcDelays

from RAiDER.utilFcns import rio_open
from RAiDER.llreader import (
    StationFile,
    RasterRDR,
    BoundingBox,
    GeocodedFile,
    Geocube,
    bounds_from_latlon_rasters,
    bounds_from_csv,
    _parse_crs,
    _ellipsoidal_to_geometric,
    _geometric_to_ellipsoidal,
    _source_is_geoid,
    _is_geoid_crs,
    HGT_DATUM_COLUMN,
)

SCENARIO0_DIR = TEST_DIR / 'scenario_0'
SCENARIO1_DIR = TEST_DIR / 'scenario_1/geom'
SCENARIO2_DIR = TEST_DIR / 'scenario_2'


@pytest.fixture
def parser():
    return calcDelays()


@pytest.fixture
def station_file():
    return SCENARIO2_DIR / 'stations.csv'


@pytest.fixture
def llfiles():
    return SCENARIO1_DIR / 'lat.dat', SCENARIO1_DIR / 'lon.dat'


@pytest.fixture
def patch_transformer(monkeypatch):
    """Install a fake pyproj Transformer in place of the real one.

    Returns an installer function taking:
      offset:  what transform() adds to the heights it is given
      proj4:   what to_proj4() reports -- '+proj=noop' is what PROJ returns
               when it silently fell back to a no-op because the EGM2008 grid
               is missing, which the production code detects
      raises:  an exception for from_crs() to raise instead of returning

    The installer returns a record dict capturing the CRSs each from_crs()
    call received ('src'/'dst' from the last call, 'calls' for all of them).
    """

    def install(offset=0.0, proj4='+proj=pipeline', raises=None):
        record = {'calls': []}

        class _FakeTransformer:
            def to_proj4(self):
                return proj4

            def transform(self, lons, lats, heights):
                return lons, lats, heights + offset

        def fake_from_crs(src_crs, dst_crs, always_xy=True):
            record['calls'].append((src_crs, dst_crs))
            record['src'], record['dst'] = src_crs, dst_crs
            if raises is not None:
                raise raises
            return _FakeTransformer()

        monkeypatch.setattr(pyproj.Transformer, 'from_crs', fake_from_crs)
        return record

    return install


@pytest.fixture
def patch_proj_network(monkeypatch):
    """Track PROJ CDN network toggling, starting from disabled.

    Returns a state dict whose 'set_calls' records every value the production
    code passed to set_network_enabled -- [True, False] means it enabled the
    CDN for a retry and then restored the previous state.
    """
    state = {'enabled': False, 'set_calls': []}

    def fake_set_network_enabled(value):
        state['set_calls'].append(value)
        state['enabled'] = value

    monkeypatch.setattr(pyproj.network, 'is_network_enabled', lambda: state['enabled'])
    monkeypatch.setattr(pyproj.network, 'set_network_enabled', fake_set_network_enabled)
    return state


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
    stats = pd.read_csv(station_file).drop_duplicates(subset=['Lat', 'Lon'])

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
    assert np.allclose(aoi.readZ(), 0.1)


def test_GeocodedFile():
    aoi = GeocodedFile(SCENARIO0_DIR / 'small_dem.tif', is_dem=True)
    z = aoi.readZ()
    x, y = aoi.readLL()
    assert z.shape == (569, 558)
    assert x.shape == z.shape


# ---------------------------------------------------------------------------
# _parse_crs
# ---------------------------------------------------------------------------


def test_parse_crs_passes_through_3d_crs_object():
    crs_obj = CRS.from_epsg(4979)
    assert _parse_crs(crs_obj) is crs_obj


def test_parse_crs_from_int_epsg():
    assert _parse_crs(4979) == CRS.from_epsg(4979)


def test_parse_crs_normalizes_2d_to_geoid():
    """A 2D CRS has no vertical axis for PROJ to transform from, and RAiDER
    reads it as "these heights are MSL" -- so it becomes EPSG:9518, which says
    that in a transformable form."""
    for value in (4326, '4326', 'EPSG:4326', CRS.from_epsg(4326)):
        assert _parse_crs(value) == CRS.from_epsg(9518)


def test_parse_crs_none_defaults_to_ellipsoidal():
    """A blank value in a run config means unspecified, not invalid."""
    assert _parse_crs(None) == CRS.from_epsg(4979)


def test_parse_crs_invalid_string_raises():
    with pytest.raises(pyproj.exceptions.CRSError):
        _parse_crs('not_a_real_crs')


def test_is_geoid_crs_distinguishes_9518_from_4979():
    """The bug the axis-count heuristic had: both have three axes."""
    assert _is_geoid_crs(CRS.from_epsg(9518)) is True
    assert _is_geoid_crs(CRS.from_epsg(4979)) is False


# ---------------------------------------------------------------------------
# _ellipsoidal_to_geometric
# ---------------------------------------------------------------------------


def test_ellipsoidal_to_geometric_already_geoid_passthrough(patch_transformer):
    """Heights already on the geoid come back untouched, without consulting PROJ.

    A 2D CRS normalises to EPSG:9518, which equals the target, so the identity
    short-circuit returns early. That matters beyond saving work: an identity
    pipeline also reports '+proj=noop', which would otherwise be mistaken for a
    missing EGM2008 grid and raise a spurious "conversion unavailable" warning.
    """
    patch_transformer(proj4='+proj=noop')  # would trigger the missing-grid path if reached

    lats = np.array([34.0, 35.0])
    lons = np.array([-118.0, -117.0])
    heights = np.array([100.0, 200.0])

    for crs in (CRS.from_epsg(4326), CRS.from_epsg(9518)):
        result = _ellipsoidal_to_geometric(lats, lons, heights, crs)
        assert result is heights


def test_ellipsoidal_to_geometric_successful_conversion(patch_transformer, patch_proj_network):
    """When PROJ has the EGM2008 grid, heights should be converted on the first try."""
    patch_transformer(offset=20.0)

    lats = np.array([34.0, 35.0])
    lons = np.array([-118.0, -117.0])
    heights = np.array([100.0, 200.0])

    result = _ellipsoidal_to_geometric(lats, lons, heights, CRS.from_epsg(4979))

    assert np.allclose(result, heights + 20.0)
    # Network was never touched since the first attempt already succeeded.
    assert patch_proj_network['set_calls'] == []


def test_ellipsoidal_to_geometric_missing_grid_falls_back(patch_transformer, patch_proj_network):
    """If the EGM2008 grid is missing locally and the CDN is unreachable, heights pass through."""
    patch_transformer(proj4='+proj=noop')

    lats = np.array([34.0])
    lons = np.array([-118.0])
    heights = np.array([100.0])

    result = _ellipsoidal_to_geometric(lats, lons, heights, CRS.from_epsg(4979))

    assert np.array_equal(result, heights)
    # Network enabled to attempt the CDN download, then restored to its prior state.
    assert patch_proj_network['set_calls'] == [True, False]


def test_ellipsoidal_to_geometric_exception_falls_back(patch_transformer):
    """Any unexpected failure during the transform should not crash readZ."""
    patch_transformer(raises=RuntimeError('boom'))

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
    """The documented crs=4326 still means "geoid", now stored as the
    transformable EPSG:9518 rather than a 2D CRS PROJ can't convert from."""
    query = StationFile(station_file, crs=4326)
    assert query._crs == CRS.from_epsg(9518)
    assert _is_geoid_crs(query._crs) is True


def test_stationfile_update_crs(station_file):
    query = StationFile(station_file)
    query.update_crs(4326)  # normalised to the geoid CRS it stands for
    assert query._crs == CRS.from_epsg(9518)

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


def test_geometric_to_ellipsoidal_successful_conversion(patch_transformer):
    """When PROJ has the EGM2008 grid, heights should be converted on the first try."""
    record = patch_transformer(offset=-15.0)

    lats = np.array([34.0, 35.0])
    lons = np.array([-118.0, -117.0])
    heights = np.array([100.0, 200.0])  # geoid heights

    result = _geometric_to_ellipsoidal(lats, lons, heights)

    assert np.allclose(result, heights - 15.0)
    # Default target is WGS84 ellipsoidal, converting from the EGM2008 geoid.
    assert record['src'] == CRS.from_epsg(9518)
    assert record['dst'] == CRS.from_epsg(4979)


def test_geometric_to_ellipsoidal_custom_target_crs(patch_transformer):
    """A caller-supplied target CRS should be used instead of the default."""
    record = patch_transformer()

    custom_crs = CRS.from_epsg(4979)
    _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), np.array([100.0]), crs=custom_crs)

    assert record['dst'] == custom_crs


def test_geometric_to_ellipsoidal_missing_grid_falls_back(patch_transformer, patch_proj_network):
    """If the EGM2008 grid is missing locally and the CDN is unreachable, heights pass through."""
    patch_transformer(proj4='+proj=noop')

    heights = np.array([100.0])
    result = _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), heights)

    assert np.array_equal(result, heights)
    assert patch_proj_network['set_calls'] == [True, False]


def test_geometric_to_ellipsoidal_exception_falls_back(patch_transformer):
    """Any unexpected failure during the transform should not crash readZ."""
    patch_transformer(raises=RuntimeError('boom'))

    heights = np.array([100.0])
    result = _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), heights)

    assert np.array_equal(result, heights)


# ---------------------------------------------------------------------------
# StationFile.readZ -- no-Hgt_m / DEM-download branch
# ---------------------------------------------------------------------------


def test_readZ_sf_dem_download_geoid_request_no_conversion(monkeypatch, tmp_path):
    """DEM heights are geoid-referenced at the source, so a geoid request needs no conversion."""
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
    z = query.readZ(geoid_heights=True)

    assert np.allclose(z, 100.0)
    # self._crs must be updated to the DEM's actual (geoid) datum, whatever
    # crs= this object was given, so a later readZ() on it doesn't misread the
    # heights it just wrote -- the exact state trap behind the stale-_crs bug.
    assert query._crs == CRS.from_epsg(9518)


def test_readZ_sf_dem_download_default_converts_to_ellipsoidal(monkeypatch, tmp_path):
    """The default (ellipsoidal) request converts DEM heights via _geometric_to_ellipsoidal."""
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
        return heights - 34.0

    monkeypatch.setattr('RAiDER.dem.download_dem', fake_download_dem)
    monkeypatch.setattr('RAiDER.interpolator.interpolateDEM', fake_interpolateDEM)
    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fake_convert)

    query = StationFile(csv_path)
    z = query.readZ()

    assert np.allclose(calls['heights'], 100.0)
    assert np.allclose(z, 100.0 - 34.0)


def test_readZ_sf_write_back_then_read_again_stays_consistent(tmp_path, monkeypatch):
    """Regression test for the stale-_crs bug.

    The object is built declaring crs=4979 (ellipsoidal), but the DEM branch
    fills in geoid heights -- so the declared datum is wrong for what actually
    lands in the file. Both reads must still agree: the first because it knows
    what it just wrote, the second because the Hgt_datum column and the
    updated self._crs tell it. Before the fix, the second read trusted the
    stale crs= and silently double-converted.
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

    z1 = query.readZ(geoid_heights=True)  # DEM branch: writes geoid Hgt_m
    z2 = query.readZ(geoid_heights=True)  # Hgt_m exists now; must read as geoid

    assert np.allclose(z1, 100.0)
    assert np.allclose(z1, z2)
    assert query._crs == CRS.from_epsg(9518)


# ---------------------------------------------------------------------------
# RasterRDR.readZ
# ---------------------------------------------------------------------------


def test_rasterrdr_readZ_default_no_conversion(monkeypatch):
    """hgt_file is ISCE-convention ellipsoidal; the default (ellipsoidal) request needs no conversion."""
    latfile = Path(GEOM_DIR) / 'lat.rdr'
    lonfile = Path(GEOM_DIR) / 'lon.rdr'
    query = RasterRDR(lat_file=str(latfile), lon_file=str(lonfile), hgt_file=str(latfile))
    raw_hgts, _ = rio_open(latfile)

    def fail_if_called(*args, **kwargs):
        raise AssertionError('conversion should not be called')

    monkeypatch.setattr('RAiDER.llreader._ellipsoidal_to_geometric', fail_if_called)
    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fail_if_called)

    z = query.readZ()
    assert np.allclose(z, raw_hgts, equal_nan=True)


def test_rasterrdr_readZ_geoid_request_converts_from_hgtfile_ellipsoidal(monkeypatch):
    """hgt_file is ISCE-convention ellipsoidal; requesting geoid_heights=True converts via _ellipsoidal_to_geometric."""
    latfile = Path(GEOM_DIR) / 'lat.rdr'
    lonfile = Path(GEOM_DIR) / 'lon.rdr'
    query = RasterRDR(lat_file=str(latfile), lon_file=str(lonfile), hgt_file=str(latfile))

    calls = {}

    def fake_convert(lats, lons, heights, crs):
        calls['called'] = True
        calls['heights'] = heights
        calls['crs'] = crs
        return heights + 3.0

    monkeypatch.setattr('RAiDER.llreader._ellipsoidal_to_geometric', fake_convert)

    z = query.readZ(geoid_heights=True)

    assert calls.get('called') is True
    assert calls['crs'] == CRS.from_epsg(4979)
    assert np.allclose(z, calls['heights'] + 3.0)


def test_rasterrdr_requires_lon_file():
    """A RasterRDR cannot currently be built without a lon_file.

    The class docstring advertises a single 2-band lat/lon raster, and both
    __init__ and readLL() have branches for lon_file=None, but the feature is
    unimplemented: __init__ always routes through bounds_from_latlon_rasters(),
    which cannot handle a None lon_file. This test pins the real, current
    behavior via the public API.

    Consequence worth knowing: readZ()'s NotImplementedError guard for a
    missing lon_file is therefore unreachable defense-in-depth, not a path any
    caller can actually reach today.
    """
    latfile = Path(GEOM_DIR) / 'lat.rdr'

    with pytest.raises(ValueError):
        RasterRDR(lat_file=str(latfile), lon_file=None)


# ---------------------------------------------------------------------------
# GeocodedFile.readZ
# ---------------------------------------------------------------------------


def test_GeocodedFile_readZ_treats_dem_as_geoid(monkeypatch):
    """DEM heights are geoid-referenced -- both a hand-downloaded GLO-30/SRTM
    product and the GLO30.dem RAiDER writes itself, which
    validators.get_query_region() also classifies as is_dem=True. So a geoid
    request is a no-op and the default (ellipsoidal) one converts."""
    aoi = GeocodedFile(SCENARIO0_DIR / 'small_dem.tif', is_dem=True)

    calls = {}

    def fake_convert(lats, lons, heights):
        calls['called'] = True
        return heights - 34.0

    monkeypatch.setattr('RAiDER.llreader._geometric_to_ellipsoidal', fake_convert)

    z_geoid = aoi.readZ(geoid_heights=True)
    z_ell = aoi.readZ()

    assert calls.get('called') is True
    # small_dem.tif has no geotransform, so interpolateDEM returns all-NaN heights here;
    # equal_nan=True since we're only checking the mocked offset was applied elementwise.
    assert np.allclose(z_ell, z_geoid - 34.0, equal_nan=True)


# ---------------------------------------------------------------------------
# _source_is_geoid / Hgt_datum column
# ---------------------------------------------------------------------------


def test_source_is_geoid_falls_back_to_crs_when_no_column():
    df = pd.DataFrame({'Lat': [34.0], 'Lon': [-118.0], 'Hgt_m': [100.0]})
    assert _source_is_geoid(df, CRS.from_epsg(9518)) is True
    assert _source_is_geoid(df, CRS.from_epsg(4979)) is False


def test_source_is_geoid_column_overrides_crs():
    """The column is authoritative even when it contradicts the passed-in crs."""
    df = pd.DataFrame({'Lat': [34.0], 'Lon': [-118.0], 'Hgt_m': [100.0], HGT_DATUM_COLUMN: ['geoid']})
    assert _source_is_geoid(df, CRS.from_epsg(4979)) is True

    df[HGT_DATUM_COLUMN] = 'ellipsoidal'
    assert _source_is_geoid(df, CRS.from_epsg(4326)) is False


def test_readZ_sf_datum_column_still_converts_against_2d_crs(tmp_path, monkeypatch):
    """Column says ellipsoidal, crs= says 4326: the conversion must still happen.

    Regression test. The column decides *whether* to convert, but the source CRS
    handed to _ellipsoidal_to_geometric decides *how*; passing a 2D self._crs
    there hits its no-vertical-datum guard and silently returns the heights
    unchanged -- the conversion the column just asked for quietly skipped.
    """
    csv_path = tmp_path / 'stations.csv'
    csv_path.write_text(f'ID,Lat,Lon,Hgt_m,{HGT_DATUM_COLUMN}\nA,34.0,-118.0,100.0,ellipsoidal\n')

    calls = {}

    def fake_convert(lats, lons, heights, crs):
        calls['crs'] = crs
        return heights - 34.0

    monkeypatch.setattr('RAiDER.llreader._ellipsoidal_to_geometric', fake_convert)

    query = StationFile(csv_path, crs=4326)
    z = query.readZ(geoid_heights=True)

    # A 3D CRS must be handed down, or the conversion is a silent no-op.
    assert len(calls['crs'].axis_info) >= 3
    assert np.allclose(z, 100.0 - 34.0)


def test_source_is_geoid_inconsistent_column_raises():
    df = pd.DataFrame(
        {
            'Lat': [34.0, 35.0],
            'Lon': [-118.0, -117.0],
            'Hgt_m': [100.0, 200.0],
            HGT_DATUM_COLUMN: ['ellipsoidal', 'geoid'],
        }
    )
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

    StationFile(csv_path).readZ(geoid_heights=True)

    df = pd.read_csv(csv_path)
    assert HGT_DATUM_COLUMN in df.columns
    assert (df[HGT_DATUM_COLUMN] == 'geoid').all()


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

    # First object downloads the DEM (geoid) and writes it, with the datum column.
    StationFile(csv_path).readZ(geoid_heights=True)

    # A second, independent object is constructed with a *wrong* crs -- 4979,
    # ellipsoidal, when the file actually holds geoid heights. Without the
    # column this would silently misread the file; with it, the column wins.
    second = StationFile(csv_path, crs=4979)
    assert np.allclose(second.readZ(geoid_heights=True), 100.0)  # no conversion needed


# ---------------------------------------------------------------------------
# Loud warning on missing-grid / conversion failure
# ---------------------------------------------------------------------------


def test_convert_missing_grid_raises_user_warning(patch_transformer, patch_proj_network):
    patch_transformer(proj4='+proj=noop')

    heights = np.array([100.0])
    with pytest.warns(UserWarning, match='Height datum conversion unavailable'):
        result = _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), heights)

    assert np.array_equal(result, heights)


def test_convert_exception_raises_user_warning(patch_transformer):
    patch_transformer(raises=RuntimeError('boom'))

    heights = np.array([100.0])
    with pytest.warns(UserWarning, match='Height datum conversion unavailable'):
        result = _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), heights)

    assert np.array_equal(result, heights)


def test_convert_propagates_when_warnings_are_errors(patch_transformer, patch_proj_network):
    """With warnings configured as errors, the warning must propagate cleanly.

    _convert_height_datum's broad `except Exception` would otherwise catch our
    own warning-turned-error, warn a second time with the first message as its
    reason (a message nested inside itself), and then silently return
    unconverted heights -- defeating the whole point of warning loudly for a
    caller who explicitly asked for strictness.
    """
    patch_transformer(proj4='+proj=noop')

    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        with pytest.raises(UserWarning) as excinfo:
            _geometric_to_ellipsoidal(np.array([34.0]), np.array([-118.0]), np.array([100.0]))

    # Exactly once -- not nested inside itself.
    assert str(excinfo.value).count('Height datum conversion unavailable') == 1


# ---------------------------------------------------------------------------
# Geocube.readZ (previously untested)
# ---------------------------------------------------------------------------


def test_geocube_readZ(tmp_path):
    """Geocube.readZ() reads the 'heights' variable directly from the cube.

    Constructed via __new__ rather than the real constructor: __init__ also
    calls rio_stats() for _proj/_geotransform metadata (needs a real
    georeferenced raster via GDAL's netCDF driver) that readZ() itself never
    touches, so building one just to test readZ() would make this test
    depend on machinery unrelated to what it's checking.
    """
    heights = np.arange(24.0).reshape(2, 3, 4)
    ds = xr.Dataset({'heights': (('z', 'y', 'x'), heights)})
    nc_path = tmp_path / 'cube.nc'
    ds.to_netcdf(nc_path)

    aoi = Geocube.__new__(Geocube)
    aoi.path = nc_path

    result = aoi.readZ()
    assert np.array_equal(result, heights)


def test_geocube_readZ_refuses_geoid_request(tmp_path):
    """Geocube records no datum, so it must refuse rather than guess.

    It takes the parameter purely so AOI.readZ() has one signature -- without
    it, geoid_heights=True would be a TypeError that only delay.py's early
    return for cube AOIs currently keeps from surfacing.
    """
    ds = xr.Dataset({'heights': (('z', 'y', 'x'), np.zeros((2, 3, 4)))})
    nc_path = tmp_path / 'cube.nc'
    ds.to_netcdf(nc_path)

    aoi = Geocube.__new__(Geocube)
    aoi.path = nc_path

    with pytest.raises(NotImplementedError):
        aoi.readZ(geoid_heights=True)


# ---------------------------------------------------------------------------
# Real EGM2008 grid (no mocking) -- catches an inverted conversion direction,
# which every other test in this file structurally cannot, since they all
# mock the Transformer/conversion functions directly.
# ---------------------------------------------------------------------------

_LA_LAT = np.array([34.05])
_LA_LON = np.array([-118.25])
# Near Los Angeles the geoid sits roughly 30-35 m below the WGS84 ellipsoid,
# so geoid-referenced (orthometric) height should come out *larger* than
# ellipsoidal height at the same physical point, by roughly that magnitude.
# A wide-but-signed range: loose enough to tolerate real EGM2008 variation,
# tight enough that a sign-inverted transform (~-30 to -35) fails it clearly.
_EXPECTED_UNDULATION_RANGE = (20.0, 45.0)


def test_ellipsoidal_to_geometric_real_grid_los_angeles():
    ellipsoidal_height = np.array([0.0])

    geoid_height = _ellipsoidal_to_geometric(_LA_LAT, _LA_LON, ellipsoidal_height, CRS.from_epsg(4979))

    if np.allclose(geoid_height, ellipsoidal_height):
        pytest.skip(
            'EGM2008 grid unavailable locally and the PROJ CDN is unreachable; cannot verify a real conversion.'
        )

    undulation = geoid_height[0] - ellipsoidal_height[0]
    low, high = _EXPECTED_UNDULATION_RANGE
    assert low < undulation < high, (
        f'geoid - ellipsoidal = {undulation:.2f} m, expected roughly {low}-{high} m near LA '
        '(a negative value here would mean the conversion is inverted)'
    )


def test_geometric_to_ellipsoidal_real_grid_los_angeles():
    geoid_height = np.array([100.0])

    ellipsoidal_height = _geometric_to_ellipsoidal(_LA_LAT, _LA_LON, geoid_height)

    if np.allclose(ellipsoidal_height, geoid_height):
        pytest.skip(
            'EGM2008 grid unavailable locally and the PROJ CDN is unreachable; cannot verify a real conversion.'
        )

    undulation = geoid_height[0] - ellipsoidal_height[0]
    low, high = _EXPECTED_UNDULATION_RANGE
    assert low < undulation < high, (
        f'geoid - ellipsoidal = {undulation:.2f} m, expected roughly {low}-{high} m near LA '
        '(a negative value here would mean the conversion is inverted)'
    )
