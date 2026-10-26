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

No network access or CDS credentials are needed. Run it from the repository root
in the RAiDER environment:

    python test/scripts/regenerate_era5_fixtures.py          # rebuild and compare
    python test/scripts/regenerate_era5_fixtures.py --write  # also overwrite the fixtures

After writing, rerun test_intersect.py and test_slant.py and update their golden
values, and run test_fixture_sanity.py.
"""

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

# (raw file, processed fixture, fixture time). A raw given as 'commit:path' is read
# from git history: the tracked LA raw was later replaced by a Mexico domain.
CASES = [
    (
        'test/weather_files/ERA-5_2019_11_17_T20_51_58.nc',
        'ERA-5_2019_11_17_T20_51_58_5S_2S_41W_37W.nc',
        dt.datetime(2019, 11, 17, 20, 51, 58, tzinfo=dt.timezone.utc),
    ),
    (
        '5f5d3807:test/weather_files/ERA-5_2020_01_30_T13_52_45.nc',
        'ERA-5_2020_01_30_T13_52_45_32N_35N_120W_115W.nc',
        dt.datetime(2020, 1, 30, 13, 52, 45, tzinfo=dt.timezone.utc),
    ),
    (
        'test/weather_files/ERA-5_2022_08_29_T17_00_01.nc',
        'ERA-5_2022_08_29_T17_00_01_69N_73N_159W_152W.nc',
        dt.datetime(2022, 8, 29, 17, 0, 1, tzinfo=dt.timezone.utc),
    ),
]

COMPARED_VARS = ('t', 'p', 'e', 'wet', 'hydro', 'wet_total', 'hydro_total')


def load_raw(source: str, work_dir: Path) -> xr.Dataset:
    """Load a raw file from the working tree, or from git history if given as 'commit:path'."""
    if ':' in source:
        path = work_dir / Path(source.split(':', 1)[1]).name
        path.write_bytes(
            subprocess.run(['git', '-C', str(REPO), 'show', source], capture_output=True, check=True).stdout
        )
    else:
        path = REPO / source
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
        for source, fixture, time in CASES:
            print(f'{fixture}  (from {source})')
            new_path = rebuild(load_raw(source, work_dir), time, work_dir / 'weather_files')
            compare(new_path, WEATHER_FILES / fixture)
            if args.write:
                shutil.copyfile(new_path, WEATHER_FILES / fixture)
                print('    written')


if __name__ == '__main__':
    main()
