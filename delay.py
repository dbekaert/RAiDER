"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently, integration is done using scipy's integrade.quad function,
although if we want to put this on the GPU, it'll need to be rewritten
in C (I think).
"""


import numpy
import scipy.integrate as integrate
import util


# The weather interface:
# point_dry_delay: x * y * z -> dry delay at that point
# point_hydrostatic_delay: x * y * z -> hydrostatic delay


class Zenith:
    """Special value indicating a look vector of "zenith"."""
    pass


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
    return integrate.quad(delay_at, 0, rnge)


def dry_delay(weather, lat, lon, height, look_vec, rnge):
    """Compute dry delay using _generic_delay."""
    def point_dry_delay(x, y, z):
        return weather.point_dry_delay(x, y, z)
    return _generic_delay(lat, lon, height, look_vec, rnge, point_dry_delay)


def hydrostatic_delay(weather, lat, lon, height, look_x, look_y, look_z, rnge):
    """Compute hydrostatic delay using _generic_delay."""
    def point_hydrostatic_delay(x, y, z):
        return weather.point_hydrostatic_delay(x, y, z)
    return _generic_delay(lat, lon, height, look_vec, rnge,
                          point_hydrostatic_delay)
