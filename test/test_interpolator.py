import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from RAiDER.interpolate import interpolate, interpolate_along_axis
from RAiDER.interpolator import RegularGridInterpolator as Interpolator
from RAiDER.interpolator import fillna3D, interp_along_axis, interpVector


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
                0, 0.84147098, 0.90929743, 0.14112001, -0.7568025, -0.95892427,
                0.5, 1.5, 2.5, 3.5, 4.5
            ]),
            6
        ),
        np.array([0.42073549, 0.87538421, 0.52520872, -0.30784124, -0.85786338])
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
    error = np.abs(filled[index] - values[index]) / \
        np.where(denom == 0, 1, denom)

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


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def test_interpolate_along_axis():
    # Rejects scalar values
    with pytest.raises(TypeError):
        interpolate_along_axis(np.array(0), np.array(0), np.array(0))

    # Rejects mismatched number of dimensions
    with pytest.raises(TypeError):
        interpolate_along_axis(np.zeros(1), np.zeros(1), np.zeros((1, 1)))
    with pytest.raises(TypeError):
        interpolate_along_axis(np.zeros(1), np.zeros((1, 1)), np.zeros(1))
    with pytest.raises(TypeError):
        interpolate_along_axis(np.zeros((1, 1)), np.zeros(1), np.zeros(1))
    with pytest.raises(TypeError):
        interpolate_along_axis(np.zeros(1), np.zeros((1, 1)), np.zeros((1, 1)))
    with pytest.raises(TypeError):
        interpolate_along_axis(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1))
    with pytest.raises(TypeError):
        interpolate_along_axis(np.zeros((1, 1)), np.zeros(1), np.zeros((1, 1)))

    # Rejects mismatched shape for points and values
    with pytest.raises(TypeError):
        interpolate_along_axis(np.zeros(1), np.zeros(2), np.zeros(1))
    with pytest.raises(TypeError):
        interpolate_along_axis(np.zeros((9, 2)), np.zeros((9, 3)), np.zeros(1))

    # Rejects bad axis
    with pytest.raises(TypeError):
        interpolate_along_axis(np.zeros(1), np.zeros(1), np.zeros(1), axis=1)
    with pytest.raises(TypeError):
        interpolate_along_axis(np.zeros(1), np.zeros(1), np.zeros(1), axis=-2)

    # Rejects bad interp_points shape
    with pytest.raises(TypeError):
        interpolate_along_axis(
            np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((3, 2))
        )
    with pytest.raises(TypeError):
        interpolate_along_axis(
            np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 3)),
            axis=0, max_threads=1
        )


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
    assert np.allclose(
        interpolate_along_axis(xs, ys, points, axis=0, max_threads=1),
        2 * points
    )


def test_interp_along_axis_1d_out_of_bounds():
    def f(x):
        return 2 * x

    xs = np.array([1, 2, 3, 4])
    ys = f(xs)

    points = np.array([0, 5])

    assert np.allclose(
        interp_along_axis(xs, points, ys, axis=0),
        np.array([np.nan, np.nan]),
        equal_nan=True
    )
    assert np.allclose(
        interpolate_along_axis(xs, ys, points, axis=0,
                               max_threads=1, fill_value=np.nan),
        np.array([np.nan, np.nan]),
        equal_nan=True
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
    assert np.allclose(
        interpolate_along_axis(xs, ys, points, axis=1),
        2 * points
    )


def test_interp_along_axis_2d_threads_edge_case():
    def f(x):
        return 2 * x

    # Max of 4 threads but 5 rows to interpolate over. Each thread will get 2
    # rows which means only 3 threads will be used
    max_threads = 4
    xs = np.array([
        [1, 2, 3, 4],
        [3, 4, 5, 6],
        [7, 8, 9, 10],
        [11, 12, 13, 14],
        [15, 16, 17, 18]
    ])
    ys = f(xs)

    points = np.array([
        [1.5, 3.1, 3.6],
        [3.5, 5.1, 5.2],
        [7.5, 9.1, 9.9],
        [11.1, 12.2, 13.3],
        [15.1, 16.2, 17.3]
    ])

    assert np.allclose(
        interp_along_axis(xs, points, ys, axis=1),
        2 * points
    )
    assert np.allclose(
        interpolate_along_axis(xs, ys, points, axis=1, max_threads=max_threads),
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
    assert np.allclose(
        interpolate_along_axis(xs, ys, points, axis=2),
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
    assert np.allclose(
        interpolate_along_axis(xs, ys, points, axis=1),
        2 * points
    )


@pytest.mark.parametrize("num_points", (7, 200, 500))
def test_interp_along_axis_3d_large(num_points):
    def f(x):
        return 2 * x

    # To scale values along axis 0 of a 3 dimensional array
    scale = np.arange(1, 101).reshape((100, 1, 1))
    axis1 = np.arange(100)
    axis2 = np.repeat(np.array([axis1]), 100, axis=0)
    xs = np.repeat(np.array([axis2]), 100, axis=0) * scale
    ys = f(xs)

    points = np.array([np.linspace(0, 99, num=num_points)]).repeat(100, axis=0)
    points = np.repeat(np.array([points]), 100, axis=0) * scale

    ans = 2 * points

    assert np.allclose(interp_along_axis(xs, points, ys, axis=2), ans)
    assert np.allclose(interpolate_along_axis(xs, ys, points, axis=2), ans)
    assert np.allclose(
        interpolate_along_axis(xs, ys, points, axis=2, assume_sorted=True), ans
    )


def test_interp_along_axis_3d_large_unsorted():
    def f(x):
        return 2 * x

    # To scale values along axis 0 of a 3 dimensional array
    scale = np.arange(1, 101).reshape((100, 1, 1))
    axis1 = np.arange(100)
    axis2 = np.repeat(np.array([axis1]), 100, axis=0)
    xs = np.repeat(np.array([axis2]), 100, axis=0) * scale
    ys = f(xs)

    points = np.array([np.linspace(0, 99, num=300)]).repeat(100, axis=0)
    points = np.repeat(np.array([points]), 100, axis=0) * scale
    points = shuffle_along_axis(points, 2)

    ans = 2 * points

    assert np.allclose(interp_along_axis(xs, points, ys, axis=2), ans)
    assert np.allclose(interpolate_along_axis(xs, ys, points, axis=2), ans)


def test_grid_dim_mismatch():
    with pytest.raises(TypeError):
        interpolate(
            points=(np.zeros((10,)), np.zeros((5,))),
            values=np.zeros((1,)),
            interp_points=np.zeros((1,))
        )


def test_basic():
    ans = interpolate(
        points=(np.array([0, 1]),),
        values=np.array([0, 1]),
        interp_points=np.array([[0.5]]),
        max_threads=1,
        assume_sorted=True
    )

    assert ans == np.array([0.5])


def test_1d_out_of_bounds():
    ans = interpolate(
        points=(np.array([0, 1]),),
        values=np.array([0, 1]),
        interp_points=np.array([[100]]),
        max_threads=1,
        assume_sorted=True
    )

    # Output is extrapolated
    assert ans == np.array([100])


def test_1d_fill_value():
    ans = interpolate(
        points=(np.array([0, 1]),),
        values=np.array([0, 1]),
        interp_points=np.array([[100]]),
        max_threads=1,
        fill_value=np.nan,
        assume_sorted=True
    )

    assert np.all(np.isnan(ans))


def test_small():
    ans = interpolate(
        points=(np.array([1, 2, 3, 4, 5, 6]),),
        values=np.array([10, 9, 30, 10, 6, 1]),
        interp_points=np.array([1.25, 2.9, 3.01, 5.7]).reshape(-1, 1)
    )

    assert ans.shape == (4,)
    assert np.allclose(ans, np.array([9.75, 27.9, 29.8, 2.5]), atol=1e-15)


def test_small_not_sorted():
    ans = interpolate(
        points=(np.array([1, 2, 3, 4, 5, 6]),),
        values=np.array([10, 9, 30, 10, 6, 1]),
        interp_points=np.array([2.9, 1.25, 5.7, 3.01]).reshape(-1, 1),
    )

    assert ans.shape == (4,)
    assert np.allclose(ans, np.array([27.9, 9.75, 2.5, 29.8]), atol=1e-15)


def test_exact_points():
    ans = interpolate(
        points=(np.array([1, 2, 3, 4, 5, 6]),),
        values=np.array([10, 9, 30, 10, 6, 1]),
        interp_points=np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    )

    assert ans.shape == (6,)
    assert np.allclose(ans, np.array([10, 9, 30, 10, 6, 1]), atol=1e-15)


def test_2d_basic():
    xs = np.array([0, 1])
    ys = np.array([0, 1])

    values = (lambda x, y: x + y)(
        *np.meshgrid(xs, ys, indexing="ij", sparse=True)
    )

    ans = interpolate(
        points=(xs, ys),
        values=values,
        interp_points=np.array([[0.5, 0.5]])
    )

    assert ans == np.array([1])


def test_2d_out_of_bounds():
    xs = np.array([0, 1])
    ys = np.array([0, 1])

    values = (lambda x, y: x + y)(
        *np.meshgrid(xs, ys, indexing="ij", sparse=True)
    )

    ans = interpolate(
        points=(xs, ys),
        values=values,
        interp_points=np.array([[100, 100]])
    )

    # Output is extrapolated
    assert ans == np.array([200])


def test_2d_fill_value():
    xs = np.array([0, 1])
    ys = np.array([0, 1])

    values = (lambda x, y: x + y)(
        *np.meshgrid(xs, ys, indexing="ij", sparse=True)
    )

    ans = interpolate(
        points=(xs, ys),
        values=values,
        interp_points=np.array([[100, 100]]),
        fill_value=np.nan
    )

    assert np.all(np.isnan(ans))


def test_2d_square_small():
    def f(x, y):
        return x ** 2 + 3 * y

    xs = np.linspace(0, 1000, 100)
    ys = np.linspace(0, 1000, 100)

    values = f(*np.meshgrid(xs, ys, indexing="ij", sparse=True))
    points = np.stack((
        np.linspace(10, 990, 5),
        np.linspace(10, 890, 5)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys),
        values=values,
        interp_points=points,
        assume_sorted=True
    )

    rgi = RegularGridInterpolator((xs, ys), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, atol=1e-15)


def test_2d_rectangle_small():
    def f(x, y):
        return x ** 2 + 3 * y

    xs = np.linspace(0, 2000, 200)
    ys = np.linspace(0, 1000, 100)

    values = f(*np.meshgrid(xs, ys, indexing="ij", sparse=True))
    points = np.stack((
        np.linspace(10, 990, 5),
        np.linspace(10, 890, 5)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys),
        values=values,
        interp_points=points,
        assume_sorted=True
    )

    rgi = RegularGridInterpolator((xs, ys), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, atol=1e-15)


def test_2d_rectangle_small_2():
    def f(x, y):
        return x ** 2 + 3 * y

    xs = np.linspace(0, 1000, 100)
    ys = np.linspace(0, 2000, 200)

    values = f(*np.meshgrid(xs, ys, indexing="ij", sparse=True))
    points = np.stack((
        np.linspace(10, 990, 5),
        np.linspace(10, 890, 5)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys),
        values=values,
        interp_points=points,
        assume_sorted=True
    )

    rgi = RegularGridInterpolator((xs, ys), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, atol=1e-15)


def test_2d_square_large():
    def f(x, y):
        return x ** 2 + 3 * y

    xs = np.linspace(-10_000, 10_000, num=1_000)
    ys = np.linspace(0, 20_000, num=1_000)

    values = f(*np.meshgrid(xs, ys, indexing="ij", sparse=True))
    num_points = 2_000_000
    points = np.stack((
        np.linspace(10, 990, num_points),
        np.linspace(10, 890, num_points)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys),
        values=values,
        interp_points=points,
        assume_sorted=True
    )

    rgi = RegularGridInterpolator((xs, ys), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, atol=1e-15)


def test_3d_basic():
    xs = np.array([0, 1])
    ys = np.array([0, 1])
    zs = np.array([0, 1])

    values = (lambda x, y, z: x + y + z)(
        *np.meshgrid(xs, ys, zs, indexing="ij", sparse=True)
    )

    ans = interpolate(
        points=(xs, ys, zs),
        values=values,
        interp_points=np.array([[0.5, 0.5, 0.5]]),
        assume_sorted=True
    )

    assert ans == np.array([1.5])


def test_3d_out_of_bounds():
    xs = np.array([0, 1])
    ys = np.array([0, 1])
    zs = np.array([0, 1])

    values = (lambda x, y, z: x + y + z)(
        *np.meshgrid(xs, ys, zs, indexing="ij", sparse=True)
    )

    ans = interpolate(
        points=(xs, ys, zs),
        values=values,
        interp_points=np.array([[100, 100, 100]]),
        assume_sorted=True
    )

    # Output is extrapolated
    assert ans == np.array([300])


def test_3d_fill_value():
    xs = np.array([0, 1])
    ys = np.array([0, 1])
    zs = np.array([0, 1])

    values = (lambda x, y, z: x + y + z)(
        *np.meshgrid(xs, ys, zs, indexing="ij", sparse=True)
    )

    ans = interpolate(
        points=(xs, ys, zs),
        values=values,
        interp_points=np.array([[100, 100, 100]]),
        fill_value=np.nan,
        assume_sorted=True
    )

    assert np.all(np.isnan(ans))


def test_3d_cube_small():
    def f(x, y, z):
        return x ** 2 + 3 * y - z

    xs = np.linspace(0, 1000, 100)
    ys = np.linspace(0, 1000, 100)
    zs = np.linspace(0, 1000, 100)

    values = f(*np.meshgrid(xs, ys, zs, indexing="ij", sparse=True))
    points = np.stack((
        np.linspace(10, 990, 5),
        np.linspace(10, 890, 5),
        np.linspace(10, 780, 5)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys, zs),
        values=values,
        interp_points=points,
        assume_sorted=True
    )

    rgi = RegularGridInterpolator((xs, ys, zs), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, 1e-15)


def test_3d_cube_small_not_sorted():
    def f(x, y, z):
        return x ** 2 + 3 * y - z

    xs = np.linspace(0, 1000, 100)
    ys = np.linspace(0, 1000, 100)
    zs = np.linspace(0, 1000, 100)

    values = f(*np.meshgrid(xs, ys, zs, indexing="ij", sparse=True))
    points = np.stack((
        np.random.uniform(10, 990, 10),
        np.random.uniform(10, 890, 10),
        np.random.uniform(10, 780, 10)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys, zs),
        values=values,
        interp_points=points,
    )

    rgi = RegularGridInterpolator((xs, ys, zs), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, 1e-15)


def test_3d_prism_small():
    def f(x, y, z):
        return x ** 2 + 3 * y - z

    xs = np.linspace(0, 2000, 200)
    ys = np.linspace(0, 1000, 100)
    zs = np.linspace(0, 1000, 50)

    values = f(*np.meshgrid(xs, ys, zs, indexing="ij", sparse=True))
    points = np.stack((
        np.linspace(10, 990, 5),
        np.linspace(10, 890, 5),
        np.linspace(10, 780, 5)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys, zs),
        values=values,
        interp_points=points,
        assume_sorted=True
    )

    rgi = RegularGridInterpolator((xs, ys, zs), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, 1e-15)


def test_3d_prism_small_2():
    def f(x, y, z):
        return x ** 2 + 3 * y - z

    xs = np.linspace(0, 2000, 100)
    ys = np.linspace(0, 1000, 200)
    zs = np.linspace(0, 1000, 50)

    values = f(*np.meshgrid(xs, ys, zs, indexing="ij", sparse=True))
    points = np.stack((
        np.linspace(10, 990, 5),
        np.linspace(10, 890, 5),
        np.linspace(10, 780, 5)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys, zs),
        values=values,
        interp_points=points,
        assume_sorted=True
    )

    rgi = RegularGridInterpolator((xs, ys, zs), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, 1e-15)


def test_3d_prism_small_3():
    def f(x, y, z):
        return x ** 2 + 3 * y - z

    xs = np.linspace(0, 2000, 50)
    ys = np.linspace(0, 1000, 200)
    zs = np.linspace(0, 1000, 100)

    values = f(*np.meshgrid(xs, ys, zs, indexing="ij", sparse=True))
    points = np.stack((
        np.linspace(10, 990, 5),
        np.linspace(10, 890, 5),
        np.linspace(10, 780, 5)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys, zs),
        values=values,
        interp_points=points,
        assume_sorted=True
    )

    rgi = RegularGridInterpolator((xs, ys, zs), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, 1e-15)


def test_3d_cube_large():
    def f(x, y, z):
        return x ** 2 + 3 * y - z

    xs = np.linspace(0, 1000, 100)
    ys = np.linspace(0, 1000, 100)
    zs = np.linspace(0, 1000, 100)

    values = f(*np.meshgrid(xs, ys, zs, indexing="ij", sparse=True))
    num_points = 2_000_000
    points = np.stack((
        np.linspace(10, 990, num_points),
        np.linspace(10, 890, num_points),
        np.linspace(10, 780, num_points)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys, zs),
        values=values,
        interp_points=points,
        assume_sorted=True
    )

    rgi = RegularGridInterpolator((xs, ys, zs), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, 1e-15)


def test_4d_basic():
    xs = np.array([0, 1])
    ys = np.array([0, 1])
    zs = np.array([0, 1])
    ws = np.array([0, 1])

    values = (lambda x, y, z, w: x + y + z + w)(
        *np.meshgrid(xs, ys, zs, ws, indexing="ij", sparse=True)
    )

    ans = interpolate(
        points=(xs, ys, zs, ws),
        values=values,
        interp_points=np.array([[0.5, 0.5, 0.5, 0.5]])
    )

    assert ans == np.array([2])


def test_4d_out_of_bounds():
    xs = np.array([0, 1])
    ys = np.array([0, 1])
    zs = np.array([0, 1])
    ws = np.array([0, 1])

    values = (lambda x, y, z, w: x + y + z + w)(
        *np.meshgrid(xs, ys, zs, ws, indexing="ij", sparse=True)
    )

    ans = interpolate(
        points=(xs, ys, zs, ws),
        values=values,
        interp_points=np.array([[100, 100, 100, 100]])
    )

    # Output is extrapolated
    assert ans == np.array([400])


def test_4d_fill_value():
    xs = np.array([0, 1])
    ys = np.array([0, 1])
    zs = np.array([0, 1])
    ws = np.array([0, 1])

    values = (lambda x, y, z, w: x + y + z + w)(
        *np.meshgrid(xs, ys, zs, ws, indexing="ij", sparse=True)
    )

    ans = interpolate(
        points=(xs, ys, zs, ws),
        values=values,
        interp_points=np.array([[100, 100, 100, 100]]),
        fill_value=np.nan
    )

    assert np.all(np.isnan(ans))


def test_4d_cube_small():
    def f(x, y, z, w):
        return x ** 2 + 3 * y - z * w

    xs = np.linspace(0, 1000, 100)
    ys = np.linspace(0, 1000, 100)
    zs = np.linspace(0, 1000, 100)
    ws = np.linspace(0, 1000, 100)

    values = f(*np.meshgrid(xs, ys, zs, ws, indexing="ij", sparse=True))
    points = np.stack((
        np.linspace(10, 990, 5),
        np.linspace(10, 890, 5),
        np.linspace(10, 780, 5),
        np.linspace(10, 670, 5)
    ), axis=-1)

    ans = interpolate(
        points=(xs, ys, zs, ws),
        values=values,
        interp_points=points,
        assume_sorted=True
    )

    rgi = RegularGridInterpolator((xs, ys, zs, ws), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, 1e-15)


def test_interpolate_wrapper():
    def f(x, y, z):
        return x ** 2 + 3 * y - z

    xs = np.linspace(0, 1000, 100)
    ys = np.linspace(0, 1000, 100)
    zs = np.linspace(0, 1000, 100)

    values = f(*np.meshgrid(xs, ys, zs, indexing="ij", sparse=True))
    points_x = np.linspace(10, 990, 5)
    points_y = np.linspace(10, 890, 5)
    points_z = np.linspace(10, 890, 5)
    points = np.stack((
        points_x,
        points_y,
        points_z
    ), axis=-1)

    interp = Interpolator((xs, ys, zs), values)
    ans = interp(points)
    ans2 = interp((points_x, points_y, points_z))
    rgi = RegularGridInterpolator((xs, ys, zs), values)
    ans_scipy = rgi(points)

    assert np.allclose(ans, ans_scipy, 1e-15)
    assert np.allclose(ans2, ans_scipy, 1e-15)
