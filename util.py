"""Geodesy-related utility functions."""


import numpy
import pyproj


def sind(x):
    """Return the sine of x when x is in degrees."""
    return numpy.sin(numpy.radians(x))


def cosd(x):
    """Return the cosine of x when x is in degrees."""
    return numpy.cos(numpy.radians(x))


def lla2ecef(lat, lon, height):
    ecef = pyproj.Proj(proj='geocent')
    lla = pyproj.Proj(proj='latlong')

    return numpy.array(pyproj.transform(lla, ecef, lon, lat, height))
