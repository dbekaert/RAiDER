"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently, integration is done using scipy's integrade.quad function,
although if we want to put this on the GPU, it'll need to be rewritten
in C (I think).
"""


import numpy
import scipy.integrate


# The weather interface:
# point_dry_delay: x * y * z -> dry delay at that point
# point_hydrostatic_delay: x * y * z -> hydrostatic delay


def _to_centered_coords(lat, lon, height):
    """Convert (lat, lon, height) to (x, y, z)."""
    # TODO: implement this
    pass


def _generic_delay(weather, lat, lon, height, look_x, look_y, look_z, rnge,
                   delay_fn):
    """Compute delay from (lat, lon, height) up to the satellite.

    Satellite position is determined from (look_x, look_y, look_z) and
    rnge, which specify the look vector and range. Integration is
    performed from (lat, lon, height) in the direction look_vec, for
    range meters.

    Delay is either dry or hydrostatic, and delay_fn queries weather for
    the appropriate value.
    """
    look_vec = numpy.array((look_x, look_y, look_z))
    unit_look_vec = look_vec / numpy.linalg.norm(look_vec)
    start_position = _to_centered_coords(lat, lon, height)
    def delay_at(t):
        (x, y, z) = start_position + look_vec * t
        return delay_fn(weather, x, y, z)
    return scipy.integrate.quad(delay_at, 0, rnge)


def dry_delay(weather, lat, lon, height, look_x, look_y, look_z, rnge):
    """Compute dry delay using _generic_delay."""
    def point_dry_delay(weather, x, y, z):
        return weather.point_dry_delay(x, y, z)
    return _generic_delay(weather, lat, lon, height, look_x, look_y, look_z,
                          rnge, point_dry_delay)


def hydrostatic_delay(weather, lat, lon, height, look_x, look_y, look_z, rnge):
    """Compute hydrostatic delay using _generic_delay."""
    def point_hydrostatic_delay(weather, x, y, z):
        return weather.point_hydrostatic_delay(x, y, z)
    return _generic_delay(weather, lat, lon, height, look_x, look_y, look_z,
                          rnge, point_hydrostatic_delay)
