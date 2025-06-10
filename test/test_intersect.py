from RAiDER.cli.raider import calcDelays
from RAiDER.utilFcns import write_yaml
import pytest
import os
import pandas as pd
import subprocess
import numpy as np

from scipy.interpolate import griddata
import rasterio

from test import TEST_DIR, WM_DIR, pushd


SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_6")


@pytest.mark.skip(reason="The lats/lons in scenario_6 are all offshore and there is no DEM")
@pytest.mark.parametrize("wm", "ERA5".split())
def test_cube_intersect(tmp_path, wm):
    # with pushd(tmp_path):
        """ Test the intersection of lat/lon files with the DEM (model height levels?) """
        outdir = os.path.join('.', "output")
        ## make the lat lon grid
        # S, N, W, E = 33.5, 34, -118.0, -117.5
        date = 20200130
        time = "13:52:45"
        # f_lat, f_lon = makeLatLonGrid([S, N, W, E], 'LA', SCENARIO_DIR, 0.25)

        ## make the run config file
        grp = {
            "date_group": {"date_start": date},
            "time_group": {"time": time, "interpolate_time": "none"},
            "weather_model": wm,
            "aoi_group": {
                "lat_file": os.path.join(SCENARIO_DIR, "lat.rdr"),
                "lon_file": os.path.join(SCENARIO_DIR, "lon.rdr"),
            },
            "runtime_group": {
                "output_directory": outdir,
                "weather_model_directory": WM_DIR,
            },
            "verbose": False,
        }

        ## generate the default run config file and overwrite it with new parms
        cfg  = write_yaml(grp, 'temp.yaml')

        ## run raider and intersect
        calcDelays([str(cfg)])

        ## hard code what it should be and check it matches
        gold = {"ERA5": 2.2787, "GMAO": np.nan, "HRRR": np.nan}

        path_delays = os.path.join(
            outdir, f'{wm}_hydro_{date}T{time.replace(":", "")}_ztd.tiff'
        )
        latf = os.path.join(SCENARIO_DIR, "lat.rdr")
        lonf = os.path.join(SCENARIO_DIR, "lon.rdr")

        hyd = rasterio.open(path_delays).read(1)
        lats = rasterio.open(latf).read(1)
        lons = rasterio.open(lonf).read(1)
        hyd = griddata(
            np.stack([lons.flatten(), lats.flatten()], axis=-1),
            hyd.flatten(),
            (-100.6, 16.15),
            method="nearest",
        )

        np.testing.assert_almost_equal(hyd, gold[wm], decimal=4)


@pytest.mark.parametrize("wm", "ERA5".split())
def test_gnss_intersect(tmp_path, wm):
    gnss_file = os.path.join(SCENARIO_DIR, "stations.csv")
    with pushd(tmp_path):
        outdir = os.path.join(tmp_path, "output")

        id = "TORP"

        date = 20200130
        time = "13:52:45"

        ## make the run config file
        grp = {
            "date_group": {"date_start": date},
            "time_group": {"time": time, "interpolate_time": "none"},
            "weather_model": wm,
            "aoi_group": {"station_file": gnss_file},
            "runtime_group": {
                "output_directory": outdir,
                "weather_model_directory": WM_DIR,
            },
            "verbose": False,
        }

        ## generate the default run config file and overwrite it with new parms
        cfg  = write_yaml(grp, 'temp.yaml')

        ## run raider and intersect
        calcDelays([str(cfg)])

        gold = {"ERA5": 2.34514, "GMAO": np.nan, "HRRR": np.nan} 
        df = pd.read_csv(
            os.path.join(outdir, f'{wm}_Delay_{date}T{time.replace(":", "")}_ztd.csv')
        )
        td = df["totalDelay"][df["ID"] == id].values

        # test for equality with golden data
        np.testing.assert_almost_equal(td.item(), gold[wm], decimal=4)
