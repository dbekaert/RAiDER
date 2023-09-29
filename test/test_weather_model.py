import datetime
import operator
import pytest

import numpy as np
from functools import reduce
from numpy import nan
from scipy.interpolate import RegularGridInterpolator as rgi
from pathlib import Path

from RAiDER.constants import _ZMIN, _ZREF
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
from RAiDER.models.hrrr import HRRR, HRRRAK, get_bounds_indices
from RAiDER.models.gmao import GMAO
from RAiDER.models.merra2 import MERRA2
from RAiDER.models.ncmr import NCMR
from RAiDER.models.customExceptions import *


_LON0 = 0
_LAT0 = 0

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
def hrrrak():
    wm = HRRRAK()
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

def test_weatherModel_basic1(model: MockWeatherModel):
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

    with pytest.raises(DatetimeOutsideRange):
        wm.checkTime(datetime.datetime(1950, 1, 1))

    wm.checkTime(datetime.datetime(2000, 1, 1))

    with pytest.raises(DatetimeOutsideRange):
        wm.checkTime(datetime.datetime.now())


def test_uniform_in_z_small(model: MockWeatherModel):
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


def test_uniform_in_z_large(model: MockWeatherModel):
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


def test_erai(erai: ERAI):
    wm = erai
    assert wm._humidityType == 'q'
    assert wm._Name == 'ERA-I'
    assert wm._valid_range[0] == datetime.datetime(1979, 1, 1)
    assert wm._valid_range[1] == datetime.datetime(2019, 8, 31)
    assert wm._proj.to_epsg() == 4326


def test_era5(era5: ERA5):
    wm = era5
    assert wm._humidityType == 'q'
    assert wm._Name == 'ERA-5'
    assert wm._valid_range[0] == datetime.datetime(1950, 1, 1)
    assert wm._proj.to_epsg() == 4326


def test_era5t(era5t: ERA5T):
    wm = era5t
    assert wm._humidityType == 'q'
    assert wm._Name == 'ERA-5T'
    assert wm._valid_range[0] == datetime.datetime(1950, 1, 1)
    assert wm._proj.to_epsg() == 4326


def test_hres(hres: HRES):
    wm = hres
    assert wm._humidityType == 'q'
    assert wm._Name == 'HRES'
    assert wm._valid_range[0] == datetime.datetime(1983, 4, 20)
    assert wm._proj.to_epsg() == 4326
    assert wm._levels == 137

    wm.update_a_b()
    assert wm._levels == 91


def test_gmao(gmao: GMAO):
    wm = gmao
    assert wm._humidityType == 'q'
    assert wm._Name == 'GMAO'
    assert wm._valid_range[0] == datetime.datetime(2014, 2, 20)
    assert wm._proj.to_epsg() == 4326


def test_merra2(merra2: MERRA2):
    wm = merra2
    assert wm._humidityType == 'q'
    assert wm._Name == 'MERRA2'
    assert wm._valid_range[0] == datetime.datetime(1980, 1, 1)
    assert wm._proj.to_epsg() == 4326


def test_hrrr(hrrr: HRRR):
    wm = hrrr
    assert wm._humidityType == 'q'
    assert wm._Name == 'HRRR'
    assert wm._valid_range[0] == datetime.datetime(2016, 7, 15)
    assert wm._proj.to_epsg() is None
    with pytest.raises(DatetimeOutsideRange):
        wm.checkTime(datetime.datetime(2010, 7, 15))
    wm.checkTime(datetime.datetime(2018, 7, 12))

    assert isinstance(wm.checkValidBounds([35, 40, -95, -90]), HRRR)

    with pytest.raises(ValueError):
        wm.checkValidBounds([45, 47, 300, 310])


def test_hrrrak(hrrrak: HRRRAK):
    wm = hrrrak
    assert wm._Name == 'HRRR-AK'
    assert wm._valid_range[0] == datetime.datetime(2018, 7, 13)

    assert isinstance(wm.checkValidBounds([45, 47, 200, 210]), HRRRAK)

    with pytest.raises(ValueError):
        wm.checkValidBounds([15, 20, 265, 270])

    with pytest.raises(DatetimeOutsideRange):
        wm.checkTime(datetime.datetime(2018, 7, 12))

    wm.checkTime(datetime.datetime(2018, 7, 15))


def test_ncmr(ncmr: NCMR):
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


def test_ztd(model: MockWeatherModel):
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


def test_get_bounds_indices():
    snwe = [-10, 10, -10, 10]
    ll = np.arange(-20, 20)
    lats, lons = np.meshgrid(ll, ll)
    xmin, xmax, ymin, ymax = get_bounds_indices(snwe, lats, lons)
    assert xmin == 10
    assert xmax == 30
    assert ymin == 10
    assert ymax == 30


def test_get_bounds_indices_2():
    snwe = [-10, 10, 170, -170]
    l = np.arange(-20, 20)
    l2 = (((np.arange(160, 200) + 180) % 360) - 180)
    lats, lons = np.meshgrid(l, l2)
    with pytest.raises(ValueError):
        get_bounds_indices(snwe, lats, lons)


def test_get_bounds_indices_2b():
    snwe = [-10, 10, 170, 190]
    l = np.arange(-20, 20)
    l2 = np.arange(160, 200)
    lats, lons = np.meshgrid(l, l2)
    xmin, xmax, ymin, ymax = get_bounds_indices(snwe, lats, lons)
    assert xmin == 10
    assert xmax == 30
    assert ymin == 10
    assert ymax == 30


def test_get_bounds_indices_3():
    snwe = [-10, 10, -10, 10]
    l = np.arange(-20, 20)
    l2 = (((np.arange(160, 200) + 180) % 360) - 180)
    lats, lons = np.meshgrid(l, l2)
    with pytest.raises(RuntimeError):
        get_bounds_indices(snwe, lats, lons)


def test_get_bounds_indices_4():
    snwe = [55, 60, 175, 185]
    l = np.arange(55, 60, 1)
    l2 = np.arange(175, 185, 1)
    lats, lons = np.meshgrid(l, l2)
    bounds_list = get_bounds_indices(snwe, lats, lons)
    assert bounds_list == (0, 4, 0, 9)


def test_hrrr_badloc(wm:hrrr=HRRR):
    wm = wm()
    wm.set_latlon_bounds([-10, 10, -10, 10])
    wm.setTime(datetime.datetime(2020, 10, 1, 0, 0, 0))
    with pytest.raises(ValueError):
        wm._fetch('dummy_filename')

@pytest.mark.long
def test_hrrrak_dl(tmp_path: Path, wm:hrrrak=HRRRAK):
    wm = wm()
    d  = tmp_path / "files"
    d.mkdir()
    fname = d / "hrrr_ak.nc"
    wm.set_latlon_bounds([65, 67, -160, -150])
    wm.setTime(datetime.datetime(2020, 12, 1, 0, 0, 0))

    wm._fetch(fname)
    assert True

@pytest.mark.long
def test_hrrrak_dl2(tmp_path: Path, wm:hrrrak=HRRRAK):
    # test the international date line crossing
    wm = wm()
    d  = tmp_path / "files"
    d.mkdir()
    fname = d / "hrrr_ak.nc"

    wm.set_latlon_bounds([50, 52, 179, -179])
    wm.setTime(datetime.datetime(2020, 12, 1, 0, 0, 0))

    wm._fetch(fname)
    assert True

