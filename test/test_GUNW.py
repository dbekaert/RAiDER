import glob
import json
import os
import shutil
import subprocess
import unittest

import jsonschema
import pytest
import rasterio as rio
import xarray as xr

import RAiDER
import RAiDER.cli.raider as raider
from RAiDER import aws
from RAiDER.cli.raider import calcDelaysGUNW


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


def test_GUNW_metadata_update(test_gunw_json_path, test_gunw_json_schema_path, tmp_path, mocker):
    """This test performs the GUNW entrypoint with bucket/prefix provided and only updates the json.
    Monkey patches the upload/download to/from s3 and the actual computation.
    """
    temp_json_path = tmp_path / 'temp.json'
    shutil.copy(test_gunw_json_path, temp_json_path)

    # We only need to make sure the json file is passed, the netcdf file name will not have
    # any impact on subsequent testing
    mocker.patch("RAiDER.aws.get_s3_file", side_effect=['foo.nc', temp_json_path])
    mocker.patch("RAiDER.aws.upload_file_to_s3")
    mocker.patch("RAiDER.aria.prepFromGUNW.main", return_value=['my_path_cfg', 'my_wavelength'])
    mocker.patch("RAiDER.cli.raider.calcDelays", return_value=['file1', 'file2'])
    mocker.patch("RAiDER.aria.calcGUNW.tropo_gunw_slc")
    mocker.patch("os.getcwd", return_value='myDir')

    iargs = ['--weather-model', 'HRES',
             '--bucket', 'myBucket',
             '--bucket-prefix', 'myPrefix']
    calcDelaysGUNW(iargs)

    metadata = json.loads(temp_json_path.read_text())
    schema = json.loads(test_gunw_json_schema_path.read_text())

    assert metadata['weather_model'] == ['HRES']
    assert (jsonschema.validate(instance=metadata, schema=schema) is None)

    assert aws.get_s3_file.mock_calls == [
        unittest.mock.call('myBucket', 'myPrefix', '.nc'),
        unittest.mock.call('myBucket', 'myPrefix', '.json'),
    ]

    RAiDER.aria.prepFromGUNW.main.assert_called_once()

    raider.calcDelays.assert_called_once_with(['my_path_cfg'])

    RAiDER.aria.calcGUNW.tropo_gunw_slc.assert_called_once_with(
        ['file1', 'file2'],
        'foo.nc',
        'my_wavelength',
        'myDir',
        True,
    )

    assert aws.upload_file_to_s3.mock_calls == [
        unittest.mock.call('foo.nc', 'myBucket', 'myPrefix'),
        unittest.mock.call(temp_json_path, 'myBucket', 'myPrefix'),
    ]
