"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently, integration is done using scipy's integrade.quad function,
although if we want to put this on the GPU, it'll need to be rewritten
in C (I think).
"""


# TODO: figure out why scipy's integrator sucks so much and how to fix
# it (on the other hand, we might have to write our own anyway to make
# it run on the GPU, so that's something to consider.


import numpy
import scipy.integrate as integrate
import util


# The weather interface:
# point_dry_delay: x * y * z -> dry delay at that point
# point_hydrostatic_delay: x * y * z -> hydrostatic delay


class Zenith:
    """Special value indicating a look vector of "zenith"."""
    pass


def _my_sucky_integrator(f, a, b):
    """Integrate f from a to b (badly)."""
    step = 1.
    num_pts = numpy.floor((b - a) / step)
    x_pts = numpy.linspace(a, b, num=numpy.ceil((b - a)/step))
    y_pts = numpy.array([f(x_pts[i]) for i in range(x_pts.size)])
    return numpy.trapz(y_pts, x_pts)


def _generic_delay(lat, lon, height, look_vec, rnge, delay_fn):
    """Compute delay from (lat, lon, height) up to the satellite.

    Satellite position is determined from (look_x, look_y, look_z) and
    rnge, which specify the look vector and range. Integration is
    performed from (lat, lon, height) in the direction look_vec, for
    range meters.

    Delay is either dry or hydrostatic, and delay_fn queries weather for
    the appropriate value.
    """
    position = util.lla2ecef(lat, lon, height)
    if look_vec is not Zenith:
        raise NotImplemented
    else:
        corrected_lv = util.lla2ecef(lat, lon, height + 1) - position
    unit_look_vec = corrected_lv / numpy.linalg.norm(corrected_lv)
    def delay_at(t):
        (x, y, z) = position + unit_look_vec * t
        return delay_fn(x, y, z)
    # TODO: figure out a good limit
    val, _ = integrate.quad(delay_at, 0, rnge, limit=1000)
    return val


def dry_delay(weather, lat, lon, height, look_vec, rnge):
    """Compute dry delay using _generic_delay."""
    return _generic_delay(lat, lon, height, look_vec, rnge,
                          weather.point_dry_delay)


def hydrostatic_delay(weather, lat, lon, height, look_vec, rnge):
    """Compute hydrostatic delay using _generic_delay."""
    return _generic_delay(lat, lon, height, look_vec, rnge,
                          weather.point_hydrostatic_delay)
