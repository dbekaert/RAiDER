import datetime
import os
import operator
import pytest

import isce3.ext.isce3 as isce
import numpy as np

from functools import reduce
from isce3.core import DateTime, TimeDelta
from numpy import nan
from scipy.interpolate import RegularGridInterpolator as rgi
from pyproj import CRS

from RAiDER.constants import _ZMIN, _ZREF
from RAiDER.delay import build_cube_ray
from RAiDER.losreader import state_to_los
from RAiDER.models.weatherModel import (
    WeatherModel,
    find_svp,
    make_raw_weather_data_filename,
    make_weather_model_filename,
)
from RAiDER.models.erai import ERAI
from RAiDER.models.era5 import ERA5
from RAiDER.models.era5t import ERA5T
from RAiDER.models.hres import HRES
from RAiDER.models.hrrr import HRRR
from RAiDER.models.gmao import GMAO
from RAiDER.models.merra2 import MERRA2
from RAiDER.models.ncmr import NCMR


_LON0 = 0
_LAT0 = 0
_OMEGA = 0.1 / (180/np.pi)


@pytest.fixture
def erai():
    wm = ERAI()
    return wm


@pytest.fixture
def era5():
    wm = ERA5()
    return wm


@pytest.fixture
def era5t():
    wm = ERA5T()
    return wm


@pytest.fixture
def hres():
    wm = HRES()
    return wm


@pytest.fixture
def gmao():
    wm = GMAO()
    return wm


@pytest.fixture
def merra2():
    wm = MERRA2()
    return wm


@pytest.fixture
def hrrr():
    wm = HRRR()
    return wm


@pytest.fixture
def ncmr():
    wm = NCMR()
    return wm


def product(iterable):
    return reduce(operator.mul, iterable, 1)


class MockWeatherModel(WeatherModel):
    """Implement abstract methods for testing."""

    def __init__(self):
        super().__init__()

        self._k1 = 1
        self._k2 = 1
        self._k3 = 1

        self._Name = "MOCK"
        self._valid_range = (datetime.datetime(1970, 1, 1), "Present")
        self._lag_time = datetime.timedelta(days=15)

    def _fetch(self, ll_bounds, time, out):
        pass

    def load_weather(self, *args, **kwargs):
        _N_Z = 32
        self._ys = np.arange(-2,3) + _LAT0
        self._xs = np.arange(-3,4) + _LON0
        self._zs = np.linspace(0, 1e5, _N_Z)
        self._t = np.ones((len(self._ys), len(self._xs), _N_Z))
        self._e = self._t.copy()
        self._e[:,3:,:] = 2

        _p = np.arange(31, -1, -1)
        self._p = np.broadcast_to(_p, self._t.shape)

        self._true_hydro_refr = np.broadcast_to(_p, (self._t.shape))
        self._true_wet_ztd = 1e-6 * 2 * np.broadcast_to(np.flip(self._zs), (self._t.shape))
        self._true_wet_ztd[:,3:] = 2 * self._true_wet_ztd[:,3:]

        self._true_hydro_ztd = np.zeros(self._t.shape)
        for layer in range(len(self._zs)):
            self._true_hydro_ztd[:,:,layer] = 1e-6 * 0.5 * (self._zs[-1] - self._zs[layer]) * _p[layer]
        
        self._true_wet_refr = 2 * np.ones(self._t.shape)
        self._true_wet_refr[:,3:] = 4
    
    def interpWet(self):
        _ifWet = rgi((self._ys, self._xs, self._zs), self._true_wet_refr)
        return _ifWet
    def interpHydro(self):
        _ifHydro = rgi((self._ys, self._xs, self._zs), self._true_hydro_refr)
        return _ifHydro


@pytest.fixture
def model():
    return MockWeatherModel()


@pytest.fixture
def setup_fake_raytracing():
    '''This sets up a fake orbit for the weather model'''
    lat0 = _LAT0 # degrees
    lon0 = _LON0
    hsat = 700000.
    omega = _OMEGA # degrees
    Nvec = 30

    t0 = DateTime("2017-02-12T01:12:30.0")

    elp = isce.core.Ellipsoid(6378137.,.0066943799901)
    look = isce.core.LookSide.Left

    sat_hgt = elp.a + hsat
    sat_lat = np.sin(np.radians(lat0))
    clat = np.cos(np.radians(lat0))

    svs = []
    for k in range(Nvec):
        dt = TimeDelta(100 * k)
        lon = lon0 + omega * dt.total_seconds()

        pos = []
        pos.append(sat_hgt * clat * np.cos(np.radians(lon)))
        pos.append(sat_hgt * clat * np.sin(np.radians(lon)))
        pos.append(sat_hgt * sat_lat)

        vel = []
        vel.append(-omega * pos[1])
        vel.append(omega * pos[0])
        vel.append(0.)

        epoch = t0 + dt

        svs.append(
            isce.core.StateVector(epoch,pos, vel)
        )

    orb = isce.core.Orbit(svs)

    return orb, look, elp, sat_hgt

def solve(R, hsat, ellipsoid, side='left'):
    # temp = 1.0 + hsat/ellipsoid.a
    # temp1 = R/ellipsoid.a
    # temp2 = R/(ellipsoid.a + hsat)
    t2 = (np.square(hsat) + np.square(R)) - np.square(ellipsoid.a)
    # cosang = 0.5 * (temp + (1.0/temp) - temp1 * temp2)
    cosang = 0.5 * t2 / (R * hsat)
    angdiff = np.arccos(cosang)
    if side=='left':
        x = _LAT0 + angdiff
    else:
        x = _LAT0 - angdiff
    return x


def test_llhs(setup_fake_raytracing, model):
    orb, look_dir, elp, sat_hgt = setup_fake_raytracing
    llhs = []
    for k in range(20):
        tinp = 5 + k * 2
        rng = 800000 + 1000 * k
        expLon = _LON0 + _OMEGA * tinp
        geocentricLat = solve(rng, sat_hgt, elp)

        xyz = [
            elp.a * np.cos(geocentricLat) * np.cos(expLon),
            elp.a * np.cos(geocentricLat) * np.sin(expLon),
            elp.a * np.sin(geocentricLat)
        ]
        llh = elp.xyz_to_lon_lat(xyz)
        llhs.append(llh)

    assert len(llhs) == 20

@pytest.mark.skip
def test_build_cube_ray(setup_fake_raytracing, model):
    orb, look_dir, elp, _ = setup_fake_raytracing
    m = model
    m.load_weather()

    ys = np.arange(-1,1) + _LAT0
    xs = np.arange(-1,1) + _LON0
    zs = np.arange(0, 1e5, 1e3)

    _Y, _X, _Z = np.meshgrid(ys, xs, zs)

    out_true = np.zeros(_Y.shape)
    t0 = orb.start_time
    tm1 = orb.end_time
    ts = np.arange(t0, tm1 + orb.time.spacing, orb.time.spacing)
    tlist = [orb.reference_epoch + isce.core.TimeDelta(dt) for dt in ts]
    ts = np.broadcast_to(tlist, (1, len(tlist))).T
    svs = np.hstack([ts, orb.position, orb.velocity])

    #TODO: Check that the look vectors are not nans
    lv, xyz = state_to_los(svs, np.stack([_Y.ravel(), _X.ravel(), _Z.ravel()], axis=-1),out="ecef")
    out = build_cube_ray(xs, ys, zs, orb, look_dir, CRS(4326), CRS(4326), [m.interpWet(), m.interpHydro()], elp=elp)
    assert out.shape == out_true.shape


def test_weatherModel_basic1(model):
    wm = model
    assert wm._zmin == _ZMIN
    assert wm._zmax == _ZREF
    assert wm.Model() == 'MOCK'

    # check some defaults
    assert wm._humidityType == 'q'

    wm.setTime(datetime.datetime(2020, 1, 1, 6, 0, 0))
    assert wm._time == datetime.datetime(2020, 1, 1, 6, 0, 0)

    wm.setTime('2020-01-01T00:00:00')
    assert wm._time == datetime.datetime(2020, 1, 1, 0, 0, 0)

    wm.setTime('19720229', fmt='%Y%m%d')  # test a leap year
    assert wm._time == datetime.datetime(1972, 2, 29, 0, 0, 0)

    with pytest.raises(RuntimeError):
        wm.checkTime(datetime.datetime(1950, 1, 1))

    wm.checkTime(datetime.datetime(2000, 1, 1))

    with pytest.raises(RuntimeError):
        wm.checkTime(datetime.datetime.now())


def test_uniform_in_z_small(model):
    # Uneven z spacing, but averages to [1, 2]
    model._zs = np.array([
        [[1., 2.],
         [0.9, 1.1]],

        [[1., 2.6],
         [1.1, 2.3]]
    ])
    model._xs = np.array([1, 2, 2])
    model._ys = np.array([2, 3, 2])
    model._p = np.arange(8).reshape(2, 2, 2)
    model._t = model._p * 2
    model._e = model._p * 3
    model._lats = model._zs  # for now just passing in dummy arrays for lats, lons
    model._lons = model._zs

    model._uniform_in_z()

    # Note that when the lower bound is exactly equal we get a value, but
    # when the upper bound is exactly equal we get the fill
    interpolated = np.array([
        [[0, nan],
         [2.5, nan]],

        [[4., 4.625],
         [nan, 6.75]]
    ])

    assert np.allclose(model._p, interpolated, equal_nan=True, rtol=0)
    assert np.allclose(model._t, interpolated * 2, equal_nan=True, rtol=0)
    assert np.allclose(model._e, interpolated * 3, equal_nan=True, rtol=0)

    assert np.allclose(model._zs, np.array([1, 2]), rtol=0)
    assert np.allclose(model._xs, np.array([1, 2]), rtol=0)
    assert np.allclose(model._ys, np.array([2, 3]), rtol=0)


def test_uniform_in_z_large(model):
    shape = (400, 500, 40)
    x, y, z = shape
    size = product(shape)

    # Uneven spacing that averages to approximately [1, 2, ..., 39, 40]
    zlevels = np.arange(1, shape[-1] + 1)
    model._zs = np.random.normal(1, 0.1, size).reshape(shape) * zlevels
    model._xs = np.empty(0)  # Doesn't matter
    model._ys = np.empty(0)  # Doesn't matter
    model._p = np.tile(np.arange(y).reshape(-1, 1) * np.ones(z), (x, 1, 1))
    model._t = model._p * 2
    model._e = model._p * 3
    model._lats = model._zs  # for now just passing in dummy arrays for lats, lons
    model._lons = model._zs

    assert model._p.shape == shape
    model._uniform_in_z()

    interpolated = np.tile(np.arange(y), (x, 1))

    assert np.allclose(np.nanmean(model._p, axis=-1),
                       interpolated, equal_nan=True, rtol=0)
    assert np.allclose(np.nanmean(model._t, axis=-1),
                       interpolated * 2, equal_nan=True, rtol=0)
    assert np.allclose(np.nanmean(model._e, axis=-1),
                       interpolated * 3, equal_nan=True, rtol=0)

    assert np.allclose(model._zs, zlevels, atol=0.05, rtol=0)


def test_mwmf():
    name = 'ERA-5'
    time = datetime.datetime(2020, 1, 1)
    ll_bounds = (-90, 90, -180, 180)
    assert make_weather_model_filename(name, time, ll_bounds) == \
        'ERA-5_2020_01_01_T00_00_00_90S_90N_180W_180E.nc'


def test_mrwmf():
    outLoc = './'
    name = 'ERA-5'
    time = datetime.datetime(2020, 1, 1)
    assert make_raw_weather_data_filename(outLoc, name, time) == \
        './ERA-5_2020_01_01_T00_00_00.nc'


def test_erai(erai):
    wm = erai
    assert wm._humidityType == 'q'
    assert wm._Name == 'ERA-I'
    assert wm._valid_range[0] == datetime.datetime(1979, 1, 1)
    assert wm._valid_range[1] == datetime.datetime(2019, 8, 31)
    assert wm._proj.to_epsg() == 4326


def test_era5(era5):
    wm = era5
    assert wm._humidityType == 'q'
    assert wm._Name == 'ERA-5'
    assert wm._valid_range[0] == datetime.datetime(1950, 1, 1)
    assert wm._proj.to_epsg() == 4326


def test_era5t(era5t):
    wm = era5t
    assert wm._humidityType == 'q'
    assert wm._Name == 'ERA-5T'
    assert wm._valid_range[0] == datetime.datetime(1950, 1, 1)
    assert wm._proj.to_epsg() == 4326


def test_hres(hres):
    wm = hres
    assert wm._humidityType == 'q'
    assert wm._Name == 'HRES'
    assert wm._valid_range[0] == datetime.datetime(1983, 4, 20)
    assert wm._proj.to_epsg() == 4326
    assert wm._levels == 137

    wm.update_a_b()
    assert wm._levels == 91


def test_gmao(gmao):
    wm = gmao
    assert wm._humidityType == 'q'
    assert wm._Name == 'GMAO'
    assert wm._valid_range[0] == datetime.datetime(2014, 2, 20)
    assert wm._proj.to_epsg() == 4326


def test_merra2(merra2):
    wm = merra2
    assert wm._humidityType == 'q'
    assert wm._Name == 'MERRA2'
    assert wm._valid_range[0] == datetime.datetime(1980, 1, 1)
    assert wm._proj.to_epsg() == 4326


def test_hrrr(hrrr):
    wm = hrrr
    assert wm._humidityType == 'q'
    assert wm._Name == 'HRRR'
    assert wm._valid_range[0] == datetime.datetime(2016, 7, 15)
    assert wm._proj.to_epsg() is None


def test_ncmr(ncmr):
    wm = ncmr
    assert wm._humidityType == 'q'
    assert wm._Name == 'NCMR'
    assert wm._valid_range[0] == datetime.datetime(2015, 12, 1)


def test_find_svp():
    t = np.arange(0, 100, 10) + 273.15
    svp_test = find_svp(t)
    svp_true = np.array([
        611.21, 1227.5981, 2337.2825, 4243.5093,
        7384.1753, 12369.2295, 20021.443, 31419.297,
        47940.574, 71305.16
    ])
    assert np.allclose(svp_test, svp_true)


def test_ztd(model):
    m = model
    m.load_weather()

    # wet refractivity will vary
    m._get_wet_refractivity()
    assert np.allclose(m._wet_refractivity, m._true_wet_refr)

    # hydro refractivity should be all the same
    m._get_hydro_refractivity()
    assert np.allclose(
        m._hydrostatic_refractivity, 
        m._true_hydro_refr,
    )

    m._getZTD()

    assert np.allclose(m._wet_ztd, m._true_wet_ztd)
    assert np.allclose(m._hydrostatic_ztd, m._true_hydro_ztd)
