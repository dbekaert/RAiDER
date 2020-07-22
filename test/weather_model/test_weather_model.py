"""
Testing the base WeatherModel class
"""
import operator
from functools import reduce

import numpy as np
import pytest
from numpy import nan

from RAiDER.models.weatherModel import WeatherModel


def product(iterable):
    return reduce(operator.mul, iterable, 1)


class MockWeatherModel(WeatherModel):
    """Implement abstract methods for testing."""

    def __init__(self):
        super().__init__()

        self._Name = "MOCK"

    def _fetch(self, lats, lons, time, out):
        pass

    def load_weather(self, *args, **kwargs):
        pass


@pytest.fixture
def model():
    return MockWeatherModel()


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

    model._uniform_in_z()

    interpolated = np.array([
        # Note that when the lower bound is exactly equal we get a value, but
        # when the upper bound is exactly equal we get the fill
        [[0, 1],
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

    assert model._p.shape == shape
    model._uniform_in_z()

    interpolated = np.tile(np.arange(y), (x, 1))

    assert np.allclose(np.nanmean(model._p, axis=-1), interpolated, equal_nan=True, rtol=0)
    assert np.allclose(np.nanmean(model._t, axis=-1), interpolated * 2, equal_nan=True, rtol=0)
    assert np.allclose(np.nanmean(model._e, axis=-1), interpolated * 3, equal_nan=True, rtol=0)

    assert np.allclose(model._zs, zlevels, atol=0.05, rtol=0)
