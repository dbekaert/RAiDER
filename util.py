"""Geodesy-related utility functions."""


import numpy


def lla2ecef(lat, lon, height):
    """Convert lat, lon, height to ECEF using the ellipsoid."""
    # This all comes straight from Wikipedia
    a = 6378137.0 # equatorial radius
    b = 6356752.3 # polar radius
    e2 = 1 - b**2/a**2 # square of first numerical eccentricity of the
                       # ellipsoid
    N = a/numpy.sqrt(1 - e2*numpy.sin(lat)**2)
    x = (N + height)*numpy.cos(lat)*numpy.cos(lon)
    y = (N + height)*numpy.cos(lat)*numpy.sin(lon)
    z = (b**2/a**2*N + height)*numpy.sin(lat)
    return numpy.array((x, y, z))
