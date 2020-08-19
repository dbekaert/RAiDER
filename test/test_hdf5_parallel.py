from test import TEST_DIR, pushd

import numpy as np
import pytest

from RAiDER.delayFcns import get_delays

# FIXME: Relying on prior setup to be performed in order for test to pass.
# This file should either by committed as test data, or set up by a fixture
# prior to the test running
POINTS_FILE = TEST_DIR / "scenario_1" / "geom" / "query_points.h5"
MODEL_FILE = TEST_DIR / "scenario_1" / "weather_files" / "ERA5_2020-01-03T23_00_00_15.75N_18.25N_-103.24E_-99.75E.h5"


@pytest.mark.skipif(
    not MODEL_FILE.exists() or not POINTS_FILE.exists(),
    reason="Will not pass until the test_scenario_*'s have run"
)
def test_get_delays_accuracy(tmp_path):
    stepSize = 15.0
    interpType = 'rgi'

    with pushd(tmp_path):
        delays_wet_1, delays_hydro_1 = get_delays(
            stepSize,
            POINTS_FILE,
            MODEL_FILE,
            interpType,
            cpu_num=1
        )

        delays_wet_4, delays_hydro_4 = get_delays(
            stepSize,
            POINTS_FILE,
            MODEL_FILE,
            interpType,
            cpu_num=4
        )
        assert np.allclose(delays_wet_1, delays_wet_4)
        assert np.allclose(delays_hydro_1, delays_hydro_4)
