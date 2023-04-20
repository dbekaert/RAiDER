import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import jsonschema
import rasterio as rio
import xarray as xr

from RAiDER import aws
from RAiDER.cli.raider import calcDelaysGUNW
import RAiDER.cli.raider as raider


@pytest.mark.isce3
@pytest.mark.parametrize('weather_model_name', ['GMAO', 'HRRR'])
def test_GUNW_update(test_dir_path, test_gunw_path, weather_model_name):
    scenario_dir = test_dir_path / 'GUNW'
    scenario_dir.mkdir(exist_ok=True, parents=True)
    orig_GUNW = test_gunw_path
    updated_GUNW = scenario_dir / orig_GUNW.name
    shutil.copy(orig_GUNW, updated_GUNW)

    cmd = f'raider.py ++process calcDelaysGUNW -f {updated_GUNW} -m {weather_model_name} -o {scenario_dir}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0

    # check the CRS and affine are written correctly
    epsg = 4326
    transform = (0.1, 0.0, -119.85, 0, -0.1, 35.55)

    group = f'science/grids/corrections/external/troposphere/{weather_model_name}/reference'
    for v in 'troposphereWet troposphereHydrostatic'.split():
        with rio.open(f'netcdf:{updated_GUNW}:{group}/{v}') as ds:
            ds.crs.to_epsg()
            assert ds.crs.to_epsg() == epsg, 'CRS incorrect'
            if weather_model_name == 'GMAO':
                assert ds.transform.almost_equals(transform), 'Affine Transform incorrect'

    with xr.open_dataset(updated_GUNW, group=group) as ds:
        for v in 'troposphereWet troposphereHydrostatic'.split():
            da = ds[v]
            if weather_model_name == 'GMAO':
                assert da.rio.transform().almost_equals(transform), 'Affine Transform incorrect'

        crs = rio.crs.CRS.from_wkt(ds['crs'].crs_wkt)
        assert crs.to_epsg() == epsg, 'CRS incorrect'

    # Clean up files
    shutil.rmtree(scenario_dir)
    os.remove('GUNW_20200130-20200124_135156.yaml')
    [os.remove(f) for f in glob.glob(f'{weather_model_name}*')]


def test_GUNW_metadata_update(test_gunw_json_path, test_gunw_json_schema_path, monkeypatch):
    """This test performs the GUNW entrypoint with bucket/prefix provided and only updates the json.
    Monkey patches the upload/download to/from s3 and the actual computation.
    """
    temp_json_path = str(test_gunw_json_path)
    temp_json_path = temp_json_path.replace('.json', '-temp.json')
    shutil.copy(test_gunw_json_path, temp_json_path)

    # Monkey patching the correction computation and s3 download/upload
    def do_nothing_factory(length_of_return_list: int = 0):
        """Returns a function with a list of specified length"""
        n = length_of_return_list
        items = ['foo'] * n if n else None

        def do_nothing(*args, **kwargs) -> list:
            return items
        return do_nothing

    def mock_s3_file(*args):
        return str(temp_json_path)

    # We only need to make sure the json file is passed, the netcdf file name will not have
    # any impact on subsequent testing
    monkeypatch.setattr(aws, "get_s3_file", mock_s3_file)
    monkeypatch.setattr(aws, "upload_file_to_s3", do_nothing_factory())
    monkeypatch.setattr(raider, "GUNW_prep", do_nothing_factory(2))
    monkeypatch.setattr(raider, "calcDelays", do_nothing_factory(2))
    monkeypatch.setattr(raider, "GUNW_calc", do_nothing_factory())

    iargs = ['--weather-model', 'HRES',
             '--bucket', 'foo',
             '--bucket-prefix', 'bar']
    calcDelaysGUNW(iargs)

    metadata = json.load(open(temp_json_path))
    schema = json.load(open(test_gunw_json_schema_path))

    assert metadata['weather_model'] == ['HRES']
    assert (jsonschema.validate(instance=metadata, schema=schema) is None)

    Path(temp_json_path).unlink()
