import datetime
import pytest

import numpy as np

from pyproj import CRS
from scipy.interpolate import RegularGridInterpolator as rgi

from RAiDER.delay import _build_cube_ray
from RAiDER.losreader import state_to_los
from RAiDER.models.weatherModel import (
    WeatherModel,
)

import isce3.ext.isce3 as isce
from isce3.core import DateTime, TimeDelta

_LON0 = 0
_LAT0 = 0
_OMEGA = 0.1 / (180/np.pi)

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
    out = _build_cube_ray(xs, ys, zs, orb, look_dir, CRS(4326), CRS(4326), [m.interpWet(), m.interpHydro()], elp=elp)
    assert out.shape == out_true.shape
