import datetime
import pytest
import numpy as np

from RAiDER.losreader import Raytracing


def test_Raytracing():
    lats = np.array([-90, 0, 0, 90])
    lons = np.array([-90, 0, 90, 180])
    hgts = np.array([-10, 0, 10, 1000])

    unit_vecs = np.array([[0,0,-1], [1,0,0], [0,1,0], [0,0,1]])

    z = Raytracing()
    with pytest.raises(RuntimeError):
        z.setPoints(lats=None)

    z.setPoints(lats=lats, lons=lons, heights = hgts)
    assert z._lats.shape == (4,)
    assert z._lats.shape == z._lons.shape
    assert np.allclose(z._heights, hgts)


def test_toa():
    lats = np.array([0, 0, 0, 0])
    lons = np.array([0, 180, 90, -90])
    hgts = np.array([0, 0, 0, 0])

    z = Raytracing()
    z.setPoints(lats=lats, lons=lons, heights=hgts)

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
