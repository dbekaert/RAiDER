import numpy as np
import pytest

from RAiDER.interpolator import (
    _interp3D, fillna3D, interp_along_axis, interpVector
)


@pytest.fixture
def grid():
    x = np.linspace(0, 10, 100)
    y = x.copy()
    z = np.arange(-1, 21)

    def f(x, y, z):
        return np.sin(x) * np.cos(y) * (0.1 * z - 5)

    values = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))

    return (x, y, z), values


def test_interpVector():
    assert np.allclose(
        interpVector(
            np.array([
                0, 1, 2, 3, 4, 5,
                0, 0.84147098,  0.90929743,  0.14112001, -0.7568025, -0.95892427,
                0.5, 1.5, 2.5, 3.5, 4.5
            ]),
            6
        ),
        np.array([0.42073549, 0.87538421, 0.52520872, -0.30784124, -0.85786338])
    )


def test_interp3D(grid):
    (x, y, z), values = grid

    interp = _interp3D(x, y, z, values, z)
    assert np.allclose(
        interp(np.array([
            [5, 5, 5], [4.5, 0.5, 15.0]
        ])),
        np.array([1.22404, 3.00252]),
        atol=0.01,
        rtol=0.01
    )


def test_fillna3D(grid):
    _, values = grid

    locations = np.array([
        [3, 2, 2],
        [0, 0, 4],
        [3, 0, 0],
        [2, 4, 3],
        [1, 0, 1],
        [3, 0, 3],
        [2, 1, 1],
        [0, 2, 1],
        [2, 1, 3],
        [3, 0, 3]
    ]).transpose()
    index = np.zeros(values.shape).astype("bool")
    index[tuple(locations)] = True

    values_with_nans = np.copy(values)
    values_with_nans[index] = np.nan
    denom = np.abs(values[index])

    filled = fillna3D(values_with_nans)
    denom = np.abs(values[index])
    error = np.abs(filled[index] - values[index]) / np.where(denom == 0, 1, denom)

    assert np.mean(error) < 0.1


def test_interp_along_axis():
    z2 = np.tile(np.arange(100)[..., np.newaxis], (5, 1, 5)).swapaxes(1, 2)
    zvals = 0.3 * z2 - 12.75

    newz = np.tile(
        np.array([1.5, 9.9, 15, 23.278, 39.99, 50.1])[..., np.newaxis],
        (5, 1, 5)
    ).swapaxes(1, 2)
    corz = 0.3 * newz - 12.75

    assert np.allclose(interp_along_axis(z2, newz, zvals, axis=2), corz)


def test_interp_along_axis_1d():
    def f(x):
        return 2 * x

    xs = np.array([1, 2, 3, 4])
    ys = f(xs)

    points = np.array([1.5, 3.1])

    assert np.allclose(
        interp_along_axis(xs, points, ys, axis=0),
        2 * points
    )


def test_interp_along_axis_2d():
    def f(x):
        return 2 * x

    xs = np.array([
        [1, 2, 3, 4],
        [3, 4, 5, 6]
    ])
    ys = f(xs)

    points = np.array([
        [1.5, 3.1, 3.6],
        [3.5, 5.1, 5.2]
    ])

    assert np.allclose(
        interp_along_axis(xs, points, ys, axis=1),
        2 * points
    )


def test_interp_along_axis_3d():
    def f(x):
        return 2 * x

    xs = np.array([
        [[1, 2, 3, 4],
         [3, 4, 5, 6]],

        [[10, 11, 12, 13],
         [21, 22, 23, 24]]
    ])
    ys = f(xs)

    points = np.array([
        [[1.5, 3.1],
         [3.5, 5.1]],

        [[10.3, 12.9],
         [22.6, 22.1]]
    ])

    assert np.allclose(
        interp_along_axis(xs, points, ys, axis=2),
        2 * points
    )


def test_interp_along_axis_3d_axis1():
    def f(x):
        return 2 * x

    xs = np.array([
        [[1, 2],
         [3, 4]],

        [[10, 11],
         [21, 22]]
    ])
    ys = f(xs)

    points = np.array([
        [[1.5, 3.1],
         [2.5, 2.1]],

        [[10.3, 12.9],
         [15, 17]]
    ])

    assert np.allclose(
        interp_along_axis(xs, points, ys, axis=1),
        2 * points
    )
