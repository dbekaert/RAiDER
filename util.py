"""Geodesy-related utility functions."""


import numpy


def sind(x):
    """Return the sine of x when x is in degrees."""
    return numpy.sin(numpy.radians(x))


def cosd(x):
    """Return the cosine of x when x is in degrees."""
    return numpy.cos(numpy.radians(x))


def lla2ecef(lat, lon, height):
    """Convert lat, lon, height to ECEF using the ellipsoid."""
    # This all comes straight from Wikipedia
    a = 6378137.0 # equatorial radius
    b = 6356752.3 # polar radius
    e2 = 1 - b**2/a**2 # square of first numerical eccentricity of the
                       # ellipsoid
    N = a/numpy.sqrt(1 - e2*sind(lat)**2)
    x = (N + height)*cosd(lat)*cosd(lon)
    y = (N + height)*cosd(lat)*sind(lon)
    z = (b**2/a**2*N + height)*sind(lat)
    return numpy.array((x, y, z))
