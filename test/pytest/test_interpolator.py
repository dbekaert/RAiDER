from RAiDER.interpolator import interp_along_axis
import numpy as np


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
