import datetime
from pathlib import Path

from RAiDER.cli.raider import read_run_config_file
from RAiDER.utilFcns import write_yaml
from test import WM, WM_DIR


def test_datelist(tmp_path: Path):
    dates = ['20200124', '20200130']
    true_dates = [
        datetime.date(2020, 1, 24),
        datetime.date(2020, 1, 30),
    ]

    dct_group = {
       'aoi_group': {'bounding_box': [28, 28.3, -116.3, -116]},
       'date_group': {'date_list': dates},
       'time_group': {'time': '00:00:00', 'interpolate_time': 'none'},
       'weather_model': WM,
       'runtime_group': {
            'output_directory': tmp_path,
            'weather_model_directory': WM_DIR
        }
    }

    cfg = write_yaml(dct_group, tmp_path / 'temp.yaml')
    param_dict = read_run_config_file(cfg)
    assert param_dict.date_group.date_list == true_dates


def test_datestep(tmp_path: Path):
    st, en, step = "20200124", "20200130", 3
    true_dates = [
        datetime.date(2020, 1, 24),
        datetime.date(2020, 1, 27),
        datetime.date(2020, 1, 30),
    ]

    dct_group = {
       'aoi_group': {'bounding_box': [28, 39, -123, -112]},
       'date_group': {'date_start': st, 'date_end': en, 'date_step': step},
       'time_group': {'time': '00:00:00', 'interpolate_time': 'none'},
       'weather_model': WM,
       'runtime_group': {
            'output_directory': tmp_path,
            'weather_model_directory': WM_DIR
        }
    }

    cfg = write_yaml(dct_group, tmp_path / 'temp.yaml')
    param_dict = read_run_config_file(cfg)
    assert param_dict.date_group.date_list == true_dates
