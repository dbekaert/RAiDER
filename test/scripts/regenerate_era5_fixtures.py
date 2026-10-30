"""Rebuild the processed ERA-5 fixtures in test/weather_files from their raw downloads.

The golden values in test_intersect.py and test_slant.py are read out of these
processed files, so any change to how ERA-5 model-level data is downloaded or
processed means they have to be rebuilt. Each one is rebuilt from the raw 2023 CDS
download it was cut from:

1. The raw is served to the real `ECMWF._get_from_cds` through a stand-in
   `cdsapi.Client`, in the layout CDS delivers today (lnsp/z at model level 1,
   t/q on all levels), so it goes through the same post-processing a user's
   download does.
2. The result is processed with `prepareWeatherModel` over the raw's full extent,
   which reproduces each fixture's grid exactly.

No network access or CDS credentials are needed, but the LA raw is read from git
history, so a shallow clone has to be deepened first (`git fetch --unshallow`).
Run it from the repository root in the RAiDER environment:

    python test/scripts/regenerate_era5_fixtures.py          # rebuild and compare
    python test/scripts/regenerate_era5_fixtures.py --write  # also overwrite the fixtures

After writing, rerun test_intersect.py and test_slant.py and update their golden
values, and run test_fixture_sanity.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from RAiDER.models.era5 import ERA5
from RAiDER.processWM import prepareWeatherModel


REPO = Path(__file__).resolve().parents[2]
WEATHER_FILES = REPO / 'test' / 'weather_files'

# (raw file, commit to read it from, processed fixture, fixture time). A commit of
# None reads the working tree; the tracked LA raw was later replaced by a Mexico
# domain, so that one is read from the commit that still had it.
CASES = [
    (
        'test/weather_files/ERA-5_2019_11_17_T20_51_58.nc',
        None,
        'ERA-5_2019_11_17_T20_51_58_5S_2S_41W_37W.nc',
        dt.datetime(2019, 11, 17, 20, 51, 58, tzinfo=dt.timezone.utc),
    ),
    (
        'test/weather_files/ERA-5_2020_01_30_T13_52_45.nc',
        '5f5d3807',
        'ERA-5_2020_01_30_T13_52_45_32N_35N_120W_115W.nc',
        dt.datetime(2020, 1, 30, 13, 52, 45, tzinfo=dt.timezone.utc),
    ),
    (
        'test/weather_files/ERA-5_2022_08_29_T17_00_01.nc',
        None,
        'ERA-5_2022_08_29_T17_00_01_69N_73N_159W_152W.nc',
        dt.datetime(2022, 8, 29, 17, 0, 1, tzinfo=dt.timezone.utc),
    ),
]

COMPARED_VARS = ('t', 'p', 'e', 'wet', 'hydro', 'wet_total', 'hydro_total')


def load_raw(raw_file: str, commit: str | None, work_dir: Path) -> xr.Dataset:
    """Load a raw file from the working tree, or from git history if a commit is given."""
    if commit is None:
        path = REPO / raw_file
    else:
        path = work_dir / Path(raw_file).name
        result = subprocess.run(
            ['git', '-C', str(REPO), 'show', f'{commit}:{raw_file}'], capture_output=True, check=False
        )
        if result.returncode != 0:
            raise SystemExit(
                f'Cannot read {raw_file} from commit {commit}: {result.stderr.decode().strip().rstrip(".")}. '
                'This needs the full git history; on a shallow clone run `git fetch --unshallow` first.'
            )
        path.write_bytes(result.stdout)
    with xr.open_dataset(path) as raw:
        return raw.load()


def fake_cds_client(raw: xr.Dataset) -> type:
    """A cdsapi.Client stand-in that serves this raw's fields in today's CDS layout."""
    coords = {
        'valid_time': raw['time'].values,
        'latitude': raw['latitude'].values,
        'longitude': raw['longitude'].values,
    }
    dims = ('valid_time', 'model_level', 'latitude', 'longitude')

    class FakeClient:
        url = 'https://cds.climate.copernicus.eu/api'

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def retrieve(self, dataset: str, params: dict, target: str) -> None:
            param = sorted(params['param'])
            if param == ['lnsp', 'z']:
                # ERA-5 archives lnsp and z at model level 1 only
                ds = xr.Dataset(
                    {v: (dims, raw[v].values[:, :1]) for v in ('lnsp', 'z')},
                    coords={**coords, 'model_level': [1]},
                )
            elif param == ['q', 't']:
                ds = xr.Dataset(
                    {v: (dims, raw[v].values) for v in ('t', 'q')},
                    coords={**coords, 'model_level': raw['level'].values},
                )
            else:
                raise ValueError(f'unexpected CDS request for {param}')
            ds.to_netcdf(target)

    return FakeClient


def rebuild(raw: xr.Dataset, time: dt.datetime, out_dir: Path) -> Path:
    """Download (from the raw) and process one weather model, returning the processed file."""
    lats = raw['latitude'].values
    lons = (raw['longitude'].values + 180) % 360 - 180
    model = ERA5()
    model.setLevelType('ml')
    model.set_wmLoc(str(out_dir))
    with patch('cdsapi.Client', fake_cds_client(raw)):
        return Path(prepareWeatherModel(model, time, [lats.min(), lats.max(), lons.min(), lons.max()]))


def compare(new_path: Path, old_path: Path) -> None:
    """Print how a rebuilt fixture differs from the tracked one."""
    with xr.open_dataset(new_path) as new, xr.open_dataset(old_path) as old:
        for coord in ('x', 'y', 'z'):
            if not np.array_equal(new[coord].values, old[coord].values):
                raise RuntimeError(f'{coord} grid differs from the tracked fixture; it cannot be replaced in place')
        for var in COMPARED_VARS:
            a, b = new[var].values, old[var].values
            diff = np.nanmax(np.abs(a - b))
            same = np.array_equal(a, b, equal_nan=True)
            print(f'    {var:11s} {"identical" if same else f"max |diff| {diff:.6g}"}')


def main() -> None:
    """Rebuild every fixture, compare it, and optionally write it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true', help='overwrite the tracked fixtures with the rebuilt files')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for raw_file, commit, fixture, time in CASES:
            print(f'{fixture}  (from {raw_file}{f" at {commit}" if commit else ""})')
            new_path = rebuild(load_raw(raw_file, commit, work_dir), time, work_dir / 'weather_files')
            compare(new_path, WEATHER_FILES / fixture)
            if args.write:
                shutil.copyfile(new_path, WEATHER_FILES / fixture)
                print('    written')


if __name__ == '__main__':
    main()
