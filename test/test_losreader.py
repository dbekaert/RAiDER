from datetime import datetime, timezone
from test import TEST_DIR

import numpy as np
import pytest

from RAiDER.losreader import read_ESA_Orbit_file, read_txt_file


@pytest.fixture
def eof_file():
    return TEST_DIR / "los_files" / "test_orbit.EOF"


@pytest.fixture
def txt_file():
    return TEST_DIR / "los_files" / "test_orbit.txt"


def test_read_orbit_file(eof_file):
    t, x, y, z, vx, vy, vz = read_ESA_Orbit_file(eof_file)

    assert np.allclose(t, [
        datetime(2020, 1, 2, 22, 59, 42, 0, timezone.utc).timestamp(),
        datetime(2020, 1, 2, 22, 59, 52, 0, timezone.utc).timestamp(),
        datetime(2020, 1, 2, 23, 0, 2, 0, timezone.utc).timestamp(),
        datetime(2020, 1, 2, 23, 0, 12, 0, timezone.utc).timestamp()
    ])
    assert np.allclose(x, [777807.283170, 753103.689876, 728381.947308, 703645.740783])
    assert np.allclose(y, [-5326950.100710, -5281833.274981, -5236088.427631, -5189720.677640])
    assert np.allclose(z, [4583342.946432, 4639190.094438, 4694513.439786, 4749306.750152])
    assert np.allclose(vx, [-2469.328881, -2471.328262, -2472.958826, -2474.221153])
    assert np.allclose(vy, [4480.112758, 4543.168347, 4605.715817, 4667.747622])
    assert np.allclose(vz, [5610.695699, 5558.629043, 5505.936161, 5452.623056])


def test_read_txt_file(txt_file):
    t, x, y, z, vx, vy, vz = read_txt_file(txt_file)

    assert np.allclose(t, [0, 10])
    assert np.allclose(x, [100, 1100])
    assert np.allclose(y, [200, 1200])
    assert np.allclose(z, [300, 1300])
    assert np.allclose(vx, [400, 1400])
    assert np.allclose(vy, [500, 1500])
    assert np.allclose(vz, [600, 1600])
