import numpy as np

# The purpose of these tests is to verify that the axis parameter for trapz is
# equivalent to calling apply_along_axis(trapz, axis).


def test_integrate_along_axis():
    y = np.array([
        [[0, 1, 2],
         [1, 2, 3]],
    ])
    x = np.array([2, 3, 4])

    for level in range(y.shape[2]):
        assert np.allclose(
            np.apply_along_axis(np.trapz, 2, y[..., level:], x=x[level:]),
            np.trapz(y[..., level:], x[level:], axis=2)
        )


def test_integrate_along_axis_2():
    y = np.array([
        [[0, 1, 2],
         [1, 2, 3]],
    ])
    x = np.linspace(1, 5, num=y.shape[2])

    for level in range(y.shape[2]):
        assert np.allclose(
            np.apply_along_axis(np.trapz, 2, y[..., level:], x=x[level:]),
            np.trapz(y[..., level:], x[level:], axis=2)
        )


def test_integrate_along_axis_large():
    y = np.random.standard_normal(100_000).reshape(100, 100, 10)
    x = np.linspace(0, 1000, num=y.shape[2])

    for level in range(y.shape[2]):
        assert np.allclose(
            np.apply_along_axis(np.trapz, 2, y[..., level:], x=x[level:]),
            np.trapz(y[..., level:], x[level:], axis=2)
        )
