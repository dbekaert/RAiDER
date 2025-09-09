from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from scipy.interpolate import griddata

from RAiDER.cli.raider import calcDelays
from RAiDER.utilFcns import write_yaml
from test import TEST_DIR, WM_DIR


SCENARIO_DIR = TEST_DIR / 'scenario_6'


@pytest.mark.skip(reason='The lats/lons in scenario_6 are all offshore and there is no DEM')
@pytest.mark.parametrize('wm', 'ERA5'.split())
def test_cube_intersect(tmp_path: Path, wm: str) -> None:
    """Test the intersection of lat/lon files with the DEM (model height levels?)."""
    outdir = tmp_path / 'output'
    ## make the lat lon grid
    # S, N, W, E = 33.5, 34, -118.0, -117.5
    date = 20200130
    time = '13:52:45'
    # f_lat, f_lon = makeLatLonGrid([S, N, W, E], 'LA', SCENARIO_DIR, 0.25)

    ## make the run config file
    grp = {
        'date_group': {'date_start': date},
        'time_group': {'time': time, 'interpolate_time': 'none'},
        'weather_model': wm,
        'aoi_group': {
            'lat_file': str(SCENARIO_DIR / 'lat.rdr'),
            'lon_file': str(SCENARIO_DIR / 'lon.rdr'),
        },
        'runtime_group': {
            'output_directory': outdir,
            'weather_model_directory': WM_DIR,
        },
        'verbose': False,
    }

    ## generate the default run config file and overwrite it with new parms
    cfg = write_yaml(grp, tmp_path / 'temp.yaml')

    ## run raider and intersect
    calcDelays([str(cfg)])

    ## hard code what it should be and check it matches
    gold = {'ERA5': 2.2787, 'GMAO': np.nan, 'HRRR': np.nan}

    path_delays = outdir / f'{wm}_hydro_{date}T{time.replace(":", "")}_ztd.tiff'
    latf = SCENARIO_DIR / 'lat.rdr'
    lonf = SCENARIO_DIR / 'lon.rdr'

    hyd = rasterio.open(path_delays).read(1)
    lats = rasterio.open(latf).read(1)
    lons = rasterio.open(lonf).read(1)
    hyd = griddata(
        np.stack([lons.flatten(), lats.flatten()], axis=-1),
        hyd.flatten(),
        (-100.6, 16.15),
        method='nearest',
    )

    np.testing.assert_almost_equal(hyd, gold[wm], decimal=4)


@pytest.mark.parametrize(
    'wm_name,gold',
    (
        ('ERA5', 2.34514),
        # Can be enabled when known-good data is added
        pytest.param('ERA5T', np.nan, marks=pytest.mark.skip),
        pytest.param('GMAO', np.nan, marks=pytest.mark.skip),
        pytest.param('MERRA2', np.nan, marks=pytest.mark.skip),
        pytest.param('NCMR', np.nan, marks=pytest.mark.skip),
        pytest.param('HRRR', np.nan, marks=pytest.mark.skip),
    ),
)
def test_gnss_intersect(tmp_path: Path, wm_name: str, gold: np.float64) -> None:
    gnss_file = SCENARIO_DIR / 'stations.csv'
    out_dir = tmp_path / 'output'

    id = 'TORP'

    date = 20200130
    time = '13:52:45'

    ## make the run config file
    grp = {
        'date_group': {'date_start': date},
        'time_group': {'time': time, 'interpolate_time': 'none'},
        'weather_model': wm_name,
        'aoi_group': {'station_file': str(gnss_file)},
        'runtime_group': {
            'output_directory': out_dir,
            'weather_model_directory': WM_DIR,
        },
        'verbose': False,
    }

    ## generate the default run config file and overwrite it with new parms
    cfg = write_yaml(grp, tmp_path / 'temp.yaml')

    ## run raider and intersect
    calcDelays([str(cfg)])

    df = pd.read_csv(out_dir / f'{wm_name}_Delay_{date}T{time.replace(":", "")}_ztd.csv')
    td = df['totalDelay'][df['ID'] == id].values

    # test for equality with golden data
    np.testing.assert_almost_equal(td.item(), gold, decimal=4)
