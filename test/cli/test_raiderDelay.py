from argparse import ArgumentParser, ArgumentTypeError
from datetime import datetime, time

from RAiDER.cli.raiderDelay import drop_nans

def test_drop_nans():
    test_d = {
        'key1': 1,
        'key2': 'string',
        'key3': None,
        'key4': {'sub_key1': 1, 'sub_key2': None},
        }
    
    out_d = drop_nans(test_d)

    assert set(out_d.keys()) == set(['key1', 'key2', 'key4'])
    
    key_list = []
    for key, value in out_d.items():
        key_list.append(key)
        if isinstance(value, dict):
            for k, v in value.items():
                key_list.append(k)

    assert set(key_list) == set(['key1', 'key2', 'key4', 'sub_key1'])
