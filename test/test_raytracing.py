import datetime
import pytest
import numpy as np

from RAiDER.losreader import Raytracing


@pytest.fixture
def rayobj():
    lats = np.array([-90, 0, 0, 90])
    lons = np.array([-90, 0, 90, 180])
    hgts = np.array([-10, 0, 10, 1000])
    z = Raytracing()
    z.setPoints(lats=lats, lons=lons, heights=hgts)
    return z

def test_Raytracing(rayobj):
    z = rayobj
    with pytest.raises(RuntimeError):
        z.setPoints(lats=None)

    assert z._lats.shape == (4,)
    assert z._lats.shape == z._lons.shape
    assert np.allclose(z._heights, np.array([-10, 0, 10, 1000]))


def test_toa(rayobj):
    z = rayobj

    # Mock xyz
    z._xyz = np.array([[6378137.0, 0.0, 0.0],
                       [-6378137.0, 0.0, 0.0],
                       [0.0, 6378137.0, 0.0],
                       [0.0, -6378137.0, 0.0]])
    z._look_vecs = np.array([[1, 0, 0],
                            [-1, 0, 0],
                            [0, 1, 0],
                            [0, -1, 0]])
    toppts = np.array([[6388137.0, 0.0, 0.0],
                       [-6388137.0, 0.0, 0.0],
                       [0.0, 6388137.0, 0.0],
                       [0.0, -6388137.0, 0.0]])

    topxyz = z.getIntersectionWithHeight(10000.)

    assert np.allclose(topxyz, toppts)
