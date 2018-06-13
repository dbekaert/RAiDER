"""Test functions for the ray-tracing suite.

These aren't meant to be automated tests or anything fancy like that,
it's just convenience for when I'm testing everything.
"""


import delay
import netcdf
import numpy


def test_weather():
    """Test the functions with some hard-coded data."""
    return netcdf.load(
            '/Users/hogenson/Desktop/APS/WRF_mexico/20070130/'
                'wrfout_d01_2007-01-30_06:00:00',
            '/Users/hogenson/Desktop/APS/WRF_mexico/20070130/'
                'wrfplev_d01_2007-01-30_06:00:00')


def test_delay(weather):
    """Calculate the delay at a particular place."""
    return delay.dry_delay(weather, 15, -100, -50, delay.Zenith, numpy.inf)
