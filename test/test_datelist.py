import datetime

from RAiDER.utilFcns import write_yaml
from test import WM, pushd
from RAiDER.cli.raider import read_run_config_file


def test_datelist(tmp_path):
    SCENARIO_DIR = tmp_path / "datelist"

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
            'output_directory': SCENARIO_DIR,
            'weather_model_directory': SCENARIO_DIR / 'weather_files'
            }
      }

    with pushd(tmp_path):
        cfg = write_yaml(dct_group, 'temp.yaml')
        param_dict = read_run_config_file(cfg)
        assert param_dict.date_group.date_list == true_dates


def test_datestep(tmp_path):
    SCENARIO_DIR = tmp_path / "scenario_5"
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
            'output_directory': SCENARIO_DIR,
            'weather_model_directory': SCENARIO_DIR / 'weather_files'
            }
      }

    with pushd(tmp_path):
        cfg = write_yaml(dct_group, 'temp.yaml')
        param_dict = read_run_config_file(cfg)
        assert param_dict.date_group.date_list == true_dates
