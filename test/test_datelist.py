import datetime
import os
import shutil
from test import TEST_DIR, WM, update_yaml
from RAiDER.cli.raider import read_template_file

def test_datelist():
    SCENARIO_DIR = os.path.join(TEST_DIR, 'datelist')
    if os.path.exists(SCENARIO_DIR):
        shutil.rmtree(SCENARIO_DIR)
    os.makedirs(SCENARIO_DIR, exist_ok=False)

    dates = ['20200124', '20200130']
    true_dates = [datetime.datetime(2020,1,24).replace(tzinfo=datetime.timezone(offset=datetime.timedelta())), 
                  datetime.datetime(2020,1,30).replace(tzinfo=datetime.timezone(offset=datetime.timedelta()))]

    dct_group = {
       'aoi_group': {'bounding_box': [28, 28.3, -116.3, -116]},
       'date_group': {'date_list': dates},
       'time_group': {'time': '00:00:00', 'interpolate_time': 'none'},
       'weather_model': WM,
       'runtime_group': {
            'output_directory': SCENARIO_DIR,
            'weather_model_directory': os.path.join(SCENARIO_DIR, 'weather_files')
            }
      }
    
    cfg  = update_yaml(dct_group, 'temp.yaml')
    param_dict = read_template_file(cfg)
    assert param_dict['date_list'] == true_dates


def test_datestep():
    SCENARIO_DIR = os.path.join(TEST_DIR, 'scenario_5')
    st, en, step = '20200124', '20200130', 3
    true_dates = [datetime.datetime(2020,1,24).replace(tzinfo=datetime.timezone(offset=datetime.timedelta())), 
                  datetime.datetime(2020,1,27).replace(tzinfo=datetime.timezone(offset=datetime.timedelta())), 
                  datetime.datetime(2020,1,30).replace(tzinfo=datetime.timezone(offset=datetime.timedelta()))]

    dct_group = {
       'aoi_group': {'bounding_box': [28, 39, -123, -112]},
       'date_group': {'date_start': st, 'date_end': en, 'date_step': step},
       'time_group': {'time': '00:00:00', 'interpolate_time': 'none'},
       'weather_model': WM,
       'runtime_group': {
            'output_directory': SCENARIO_DIR,
            'weather_model_directory': os.path.join(SCENARIO_DIR, 'weather_files')
            }
      }
    
    cfg  = update_yaml(dct_group, 'temp.yaml')
    param_dict = read_template_file(cfg)
    assert param_dict['date_list'] == true_dates