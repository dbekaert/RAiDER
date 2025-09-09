from pathlib import Path

import numpy as np
import xarray as xr
from pytest_mock import MockerFixture

from RAiDER.cli.raider import calcDelays
from RAiDER.utilFcns import write_yaml
from test import WM_DIR


def test_hrrr_ztd(tmp_path: Path, data_for_hrrr_ztd: Path, mocker: MockerFixture) -> None:
    mocker.patch('RAiDER.processWM.prepareWeatherModel', side_effect=[str(data_for_hrrr_ztd)])

    dct_group = {
        'aoi_group': {'bounding_box': [36, 37, -92, -91]},
        'date_group': {'date_start': 20200101},
        'time_group': {'time': '12:00:00', 'interpolate_time': 'none'},
        'weather_model': 'HRRR',
        'height_group': {'height_levels': [0, 50, 100, 500, 1000]},
        'runtime_group': {
            'output_directory': tmp_path,
            'weather_model_directory': WM_DIR,
        },
    }

    cfg = write_yaml(dct_group, tmp_path / 'temp.yaml')
    calcDelays([str(cfg)])

    new_data = xr.load_dataset(tmp_path / 'HRRR_tropo_20200101T120000_ztd.nc')
    new_data1 = new_data.sel(x=-91.84, y=36.84, z=0, method='nearest')
    golden_data = 2.2622863, 0.0361021  # hydro|wet

    np.testing.assert_almost_equal(golden_data[0], new_data1['hydro'].data)
    np.testing.assert_almost_equal(golden_data[1], new_data1['wet'].data)
