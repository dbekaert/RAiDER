#!/usr/bin/env python3
"""
Scenario 9: HRES model-level integration regression test for PR #820.

Verifies the layout-keyed reader and hydrostatic height reconstruction
on a single datetime and bounding box using cached HRES data.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

try:
    from RAIDER.models.hres import HRES
except ImportError:
    from RAiDER.models.hres import HRES


# --- Configuration: Datetime & Bounding Box ---
DATE_STR = "2016-06-29"
TIME_STR = "00:00:00"
BBOX = [34.0, 35.0, -119.0, -118.0]  # [S, N, W, E]

# Local file path in scenario_9 directory
SCENARIO_DIR = Path(__file__).resolve().parent
date_tag = DATE_STR.replace("-", "_")
time_tag = TIME_STR.replace(":", "_")
RAW_HRES_FILE = SCENARIO_DIR / "HRES" / f"HRES_{date_tag}_T{time_tag}.nc"

def test_hres_model_level():
    """
    Verify HRES model-level height reconstruction 
    and surface geopotential integration.
    """
    if not RAW_HRES_FILE.exists():
        pytest.skip(f"Raw HRES test file not found at: {RAW_HRES_FILE}")

    # 1. Initialize HRES weather model
    model = HRES()
    model.setLevelType("ml")  # Exercise ECMWF model-level code path
    model.set_latlon_bounds(BBOX)

    # 2. Set UTC-aware model time (required for HRES level cutoff check)
    target_dt = dt.datetime(2016, 6, 29, 0, 0, 0, tzinfo=dt.timezone.utc)
    if hasattr(model, "setTime"):
        try:
            model.setTime(target_dt)
        except Exception:
            pass
    model._time = target_dt

    # 3. Load weather data (invokes _makeDataCubes & _load_model_level)
    model.load_weather(RAW_HRES_FILE)

    # 4. Assertions and Integrity Checks
    assert model._zs is not None, "Height cube (_zs) is None!"
    assert np.all(
        np.isfinite(model._zs)
    ), "Non-finite values detected in height cube (_zs)!"

    assert model._p is not None, "Pressure cube (_p) is None!"
    assert np.all(
        np.isfinite(model._p)
    ), "Non-finite values detected in pressure cube (_p)!"

    assert np.all(
        np.diff(model._zs, axis=2) > 0
    ), "Heights are not monotonically increasing upward!"


if __name__ == "__main__":
    print("=== Running HRES Test for PR #820 ===")
    print(f"Target Datetime : {DATE_STR} {TIME_STR}")
    print(f"Bounding Box    : SNWE = {BBOX}")
    print(f"Raw File Path   : {RAW_HRES_FILE}")

    test_hres_model_level()
    print("\n>>> [PASS] All HRES checks passed successfully! <<<\n")
