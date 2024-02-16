import glob
import json
import os
import shutil
import unittest
from pathlib import Path

import eof.download
import jsonschema
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
import xarray as xr

import RAiDER
import RAiDER.cli.raider as raider
import RAiDER.s1_azimuth_timing
from RAiDER import aws
from RAiDER.aria.prepFromGUNW import (
    check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation,
    check_weather_model_availability,
)
from RAiDER.cli.raider import calcDelaysGUNW
from RAiDER.models.customExceptions import *


def compute_transform(lats, lons):
    """ Hand roll an affine transform from lat/lon coords """
    a = lons[1] - lons[0]  # lon spacing
    b = 0
    c = lons[0] - a/2  # lon start, adjusted by half a grid cell
    d = 0
    e = lats[1] - lats[0]
    f = lats[0] - e/2
    return (a, b, c, d, e, f)


@pytest.mark.isce3
@pytest.mark.parametrize('weather_model_name', ['MERRA2'])
def test_GUNW_dataset_update(test_dir_path, test_gunw_path_factory, weather_model_name,
                             weather_model_dict_for_gunw_integration_test, mocker):
    """The GUNW from test gunw factor is:

    S1-GUNW-D-R-071-tops-20200130_20200124-135156-34956N_32979N-PP-913f-v2_0_4.nc

    Therefore relevant GMAO datetimes are
    12 pm and 3 pm (in that order)
    """
    scenario_dir = test_dir_path / 'GUNW'
    scenario_dir.mkdir(exist_ok=True, parents=True)
    orig_GUNW = test_gunw_path_factory()
    updated_GUNW = scenario_dir / orig_GUNW.name
    shutil.copy(orig_GUNW, updated_GUNW)

    iargs = ['--weather-model', weather_model_name,
             '--file', str(updated_GUNW),
             '-interp', 'center_time']

    side_effect = weather_model_dict_for_gunw_integration_test[weather_model_name]
    # RAiDER needs strings for paths
    side_effect = list(map(str, side_effect))
    mocker.patch('RAiDER.processWM.prepareWeatherModel',
                 side_effect=side_effect)
    calcDelaysGUNW(iargs)

    # check the CRS and affine are written correctly
    epsg = 4326

    group = f'science/grids/corrections/external/troposphere/{weather_model_name}/reference'

    with xr.open_dataset(updated_GUNW, group=group) as ds:
        for v in 'troposphereWet troposphereHydrostatic'.split():
            da = ds[v]
            lats, lons = da.latitudeMeta.to_numpy(), da.longitudeMeta.to_numpy()
            transform = compute_transform(lats, lons)
            assert da.rio.transform().almost_equals(transform), 'Affine Transform incorrect'

        crs = rio.crs.CRS.from_wkt(ds['crs'].crs_wkt)
        assert crs.to_epsg() == epsg, 'CRS incorrect'

    for v in 'troposphereWet troposphereHydrostatic'.split():
        with rio.open(f'netcdf:{updated_GUNW}:{group}/{v}') as ds:
            ds.crs.to_epsg()
            assert ds.crs.to_epsg() == epsg, 'CRS incorrect'
            assert ds.transform.almost_equals(transform), 'Affine Transform incorrect'

    # Clean up files
    shutil.rmtree(scenario_dir)
    os.remove('GUNW_20200130-20200124_135156.yaml')
    [os.remove(f) for f in glob.glob(f'{weather_model_name}*')]


def test_GUNW_hyp3_metadata_update(test_gunw_json_path, test_gunw_json_schema_path, tmp_path, mocker):
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
    mocker.patch('RAiDER.aria.prepFromGUNW.check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation',
                 side_effect=[True])
    mocker.patch("RAiDER.aria.prepFromGUNW.check_weather_model_availability", return_value=True)
    mocker.patch("RAiDER.cli.raider.calcDelays", return_value=['file1', 'file2'])
    mocker.patch("RAiDER.aria.calcGUNW.tropo_gunw_slc")

    iargs = ['--weather-model', 'HRES',
             '--bucket', 'myBucket',
             '--bucket-prefix', 'myPrefix']
    calcDelaysGUNW(iargs)

    metadata = json.loads(temp_json_path.read_text())
    schema = json.loads(test_gunw_json_schema_path.read_text())

    assert metadata['metadata']['weather_model'] == ['HRES']
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
    )

    assert aws.upload_file_to_s3.mock_calls == [
        unittest.mock.call('foo.nc', 'myBucket', 'myPrefix'),
        unittest.mock.call(temp_json_path, 'myBucket', 'myPrefix'),
    ]


@pytest.mark.parametrize('weather_model_name', ['HRRR'])
def test_azimuth_timing_interp_against_center_time_interp(weather_model_name: str,
                                                          tmp_path: Path,
                                                          gunw_azimuth_test: Path,
                                                          orbit_dict_for_azimuth_time_test: dict[str],
                                                          weather_model_dict_for_azimuth_time_test,
                                                          weather_model_dict_for_center_time_test,
                                                          mocker):
    """This test shows that the azimuth timing interpolation does not deviate from
    the center time by more than 1 mm for the HRRR model. This is expected since the model times are
    6 hours apart and a the azimuth time is changing the interpolation weights for a given pixel at the order
    of seconds and thus these two approaches are quite similar.

    Effectively, this mocks the following CL script and then compares the output:

    ```
    cmd = f'raider.py ++process calcDelaysGUNW -f {out_path_0} -m {weather_model_name} -interp center_time'
    subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)

    cmd = f'raider.py ++process calcDelaysGUNW -f {out_path_1} -m {weather_model_name} -interp azimuth_time_grid'
    subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    ```

    Getting weather model file names requires patience. We mock API requests.
    For HRRR and center time, here are the calls:

    ```
    wfile_0 = prepareWeatherModel(model, datetime.datetime(2021, 7, 23, 1, 0), [33.45, 35.45, -119.15, -115.95])
    wfile_1 = prepareWeatherModel(model, datetime.datetime(2021, 7, 23, 2, 0), [33.45, 35.45, -119.15, -115.95])
    wfile_2 = prepareWeatherModel(model, datetime.datetime(2021, 7, 11, 1, 0), [33.45, 35.45, -119.15, -115.95])
    wfile_3 = prepareWeatherModel(model, datetime.datetime(2021, 7, 11, 2, 0), [33.45, 35.45, -119.15, -115.95])
    ```

    And similarly for the azimuth time:

    ```
    wfile_0 = prepareWeatherModel(model, datetime.datetime(2021, 7, 23, 2, 0), [33.45, 35.45, -119.15, -115.95])
    wfile_1 = prepareWeatherModel(model, datetime.datetime(2021, 7, 23, 1, 0), [33.45, 35.45, -119.15, -115.95])
    wfile_3 = prepareWeatherModel(model, datetime.datetime(2021, 7, 11, 2, 0), [33.45, 35.45, -119.15, -115.95])
    wfile_4 = prepareWeatherModel(model, datetime.datetime(2021, 7, 11, 1, 0), [33.45, 35.45, -119.15, -115.95])
    ```

    Note for azimuth time they are acquired in order of proximity to acq as opposed to chronological.

    For the test GUNW, we have the following input granules:

    ref: S1B_IW_SLC__1SDV_20210723T014947_20210723T015014_027915_0354B4_B3A9
    sec: S1B_IW_SLC__1SDV_20210711T015011_20210711T015038_027740_034F80_376C,
         S1B_IW_SLC__1SDV_20210711T014922_20210711T014949_027740_034F80_859D
    """

    out_0 = gunw_azimuth_test.name.replace('.nc', '__ct_interp.nc')
    out_1 = gunw_azimuth_test.name.replace('.nc', '__az_interp.nc')

    out_path_0 = shutil.copy(gunw_azimuth_test, tmp_path / out_0)
    out_path_1 = shutil.copy(gunw_azimuth_test, tmp_path / out_1)

    # In the loop here: https://github.com/dbekaert/RAiDER/blob/
    # f77af9ce2d3875b00730603305c0e92d6c83adc2/tools/RAiDER/cli/raider.py#L233-L237
    # Which reads the datelist from the YAML
    # We note that reference scene is processed *first* and then secondary
    # as yaml is created using these:
    # https://github.com/dbekaert/RAiDER/blob/
    # f77af9ce2d3875b00730603305c0e92d6c83adc2/tools/RAiDER/aria/prepFromGUNW.py#L151-L200

    # For prepGUNW
    side_effect = [
        # center-time
        [Path(orbit_dict_for_azimuth_time_test['reference'])],
        # azimuth-time
        [Path(orbit_dict_for_azimuth_time_test['reference'])],
    ]
    mocker.patch('eof.download.download_eofs',
                 side_effect=side_effect)

    # These outputs are not needed since the orbits are specified above
    mocker.patch('RAiDER.s1_azimuth_timing.get_slc_id_from_point_and_time',
                 side_effect=[
                              # Azimuth time
                              ['reference_slc_id'],
                              # using two "dummy" ids to mimic GUNW sec granules
                              # See docstring
                              ['secondary_slc_id', 'secondary_slc_id'],
                             ])

    mocker.patch(
        'RAiDER.s1_azimuth_timing.get_orbits_from_slc_ids',
        side_effect=[
            # For azimuth time
            [Path(orbit_dict_for_azimuth_time_test['reference'])],
            [Path(orbit_dict_for_azimuth_time_test['secondary']), Path(orbit_dict_for_azimuth_time_test['secondary'])],
        ]
    )

    side_effect = (weather_model_dict_for_center_time_test[weather_model_name] +
                   weather_model_dict_for_azimuth_time_test[weather_model_name])
    # RAiDER needs strings for paths
    side_effect = list(map(str, side_effect))
    mocker.patch('RAiDER.processWM.prepareWeatherModel',
                 side_effect=side_effect)
    iargs_0 = [
               '--file', str(out_path_0),
               '--weather-model', weather_model_name,
               '-interp', 'center_time'
               ]
    calcDelaysGUNW(iargs_0)

    iargs_1 = [
               '--file', str(out_path_1),
               '--weather-model', weather_model_name,
               '-interp', 'azimuth_time_grid'
               ]
    calcDelaysGUNW(iargs_1)

    # Calls 4 times for azimuth time and 4 times for center time
    assert RAiDER.processWM.prepareWeatherModel.call_count == 8
    # Only calls once each ref and sec list of slcs
    assert RAiDER.s1_azimuth_timing.get_orbits_from_slc_ids.call_count == 2
    # Only calls for azimuth timing: once for ref and sec
    assert RAiDER.s1_azimuth_timing.get_slc_id_from_point_and_time.call_count == 2
    # Once for center-time and azimuth-time each
    assert eof.download.download_eofs.call_count == 2

    for ifg_type in ['reference', 'secondary']:
        for var in ['troposphereHydrostatic', 'troposphereWet']:
            group = f'science/grids/corrections/external/troposphere/{weather_model_name}/{ifg_type}'
            with xr.open_dataset(out_path_0, group=group) as ds:
                da_0 = ds[var]
            with xr.open_dataset(out_path_1, group=group) as ds:
                da_1 = ds[var]
            # diff * wavelength / (4 pi) transforms to meters; then x 1000 to mm
            abs_diff_mm = np.abs((da_1 - da_0).data) * 0.055465761572122574 / (4 * np.pi) * 1_000
            # Differences in mm are bounded by 1
            assert np.nanmax(abs_diff_mm) < 1


@pytest.mark.parametrize('weather_model_name', ['MERRA2', 'HRRR', 'HRES', 'ERA5', 'ERA5T'])
def test_check_weather_model_availability(test_gunw_path_factory, weather_model_name, mocker):
    # Should be True for all weather models
    # S1-GUNW-D-R-071-tops-20200130_20200124-135156-34956N_32979N-PP-913f-v2_0_4.nc
    test_gunw_path = test_gunw_path_factory()
    assert check_weather_model_availability(test_gunw_path, weather_model_name)

    # Let's mock an earlier date for some models
    mocker.patch("RAiDER.aria.prepFromGUNW.get_acq_time_from_slc_id", side_effect=[pd.Timestamp('2015-01-01'),
                                                                                   pd.Timestamp('2014-01-01')])
    cond = check_weather_model_availability(test_gunw_path, weather_model_name)
    if weather_model_name in ['HRRR', 'MERRA2']:
        cond = not cond
    assert cond


@pytest.mark.parametrize('weather_model_name', ['MERRA2', 'HRRR'])
def test_check_weather_model_availability_over_alaska(test_gunw_path_factory, weather_model_name, mocker):
    # Should be True for all weather models
    # S1-GUNW-D-R-059-tops-20230320_20220418-180300-00179W_00051N-PP-c92e-v2_0_6.nc
    test_gunw_path = test_gunw_path_factory(location='alaska')
    assert check_weather_model_availability(test_gunw_path, weather_model_name)

    # Let's mock an earlier date
    mocker.patch("RAiDER.aria.prepFromGUNW.get_acq_time_from_slc_id", side_effect=[pd.Timestamp('2017-01-01'),
                                                                                   pd.Timestamp('2016-01-01')])
    cond = check_weather_model_availability(test_gunw_path, weather_model_name)
    if weather_model_name == 'HRRR':
        cond = not cond
    assert cond


@pytest.mark.parametrize('weather_model_name', ['HRRR', 'MERRA2'])
@pytest.mark.parametrize('location', ['california-t71', 'alaska'])
def test_weather_model_availability_integration_using_valid_range(location,
                                                                  test_gunw_path_factory,
                                                                  tmp_path,
                                                                  weather_model_name,
                                                                  mocker):
    temp_json_path = tmp_path / 'temp.json'
    test_gunw_path = test_gunw_path_factory(location=location)
    shutil.copy(test_gunw_path, temp_json_path)

    # We will pass the test GUNW to the workflow
    mocker.patch("RAiDER.aws.get_s3_file", side_effect=[test_gunw_path, 'foo.json'])
    mocker.patch("RAiDER.aws.upload_file_to_s3")

    # Have another test for checking the actual files - we are only checking for valid
    if weather_model_name == 'HRRR':
        mocker.patch('RAiDER.aria.prepFromGUNW.check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation',
                     side_effect=[True])
    # These are outside temporal availability of GMAO and HRRR
    ref_date, sec_date = pd.Timestamp('2015-01-01'), pd.Timestamp('2014-01-01')
    mocker.patch("RAiDER.aria.prepFromGUNW.get_acq_time_from_slc_id", side_effect=[ref_date, sec_date])
    # Don't specify side-effects or return values, because never called
    mocker.patch("RAiDER.aria.prepFromGUNW.main")
    mocker.patch("RAiDER.cli.raider.calcDelays")
    mocker.patch("RAiDER.aria.calcGUNW.tropo_gunw_slc")

    iargs = ['--weather-model', weather_model_name,
             '--bucket', 'myBucket',
             '--bucket-prefix', 'myPrefix']
    out = calcDelaysGUNW(iargs)
    # Check it returned None
    assert out is None

    # Check these functions were not called
    RAiDER.cli.raider.calcDelays.assert_not_called()
    RAiDER.aria.prepFromGUNW.main.assert_not_called()
    RAiDER.aria.calcGUNW.tropo_gunw_slc.assert_not_called()


@pytest.mark.parametrize('weather_model_name', ['HRRR'])
@pytest.mark.parametrize('interp_method', ['center_time', 'azimuth_time_grid'])
def test_provenance_metadata_for_tropo_group(weather_model_name: str,
                                             tmp_path: Path,
                                             gunw_azimuth_test: Path,
                                             orbit_dict_for_azimuth_time_test: dict[str],
                                             weather_model_dict_for_azimuth_time_test,
                                             weather_model_dict_for_center_time_test,
                                             interp_method,
                                             mocker):
    """
    Same mocks as `test_azimuth_timing_interp_against_center_time_interp` above.
    """

    out = gunw_azimuth_test.name.replace('.nc', '__ct_interp.nc')

    out_path = shutil.copy(gunw_azimuth_test, tmp_path / out)

    if interp_method == 'azimuth_time_grid':
        # For prepGUNW
        side_effect = [
             # center-time
            [Path(orbit_dict_for_azimuth_time_test['reference'])],
             # azimuth-time
            [Path(orbit_dict_for_azimuth_time_test['reference'])],
        ]
        mocker.patch('eof.download.download_eofs',
                     side_effect=side_effect)

        # These outputs are not needed since the orbits are specified above
        mocker.patch('RAiDER.s1_azimuth_timing.get_slc_id_from_point_and_time',
                     side_effect=[
                                 # Azimuth time
                                 ['reference_slc_id'],
                                 # using two "dummy" ids to mimic GUNW sec granules
                                 # See docstring
                                 ['secondary_slc_id', 'secondary_slc_id'],
                                 ])

        mocker.patch(
            'RAiDER.s1_azimuth_timing.get_orbits_from_slc_ids',
            side_effect=[
                # For azimuth time
                [Path(orbit_dict_for_azimuth_time_test['reference'])],
                [Path(orbit_dict_for_azimuth_time_test['secondary']), Path(orbit_dict_for_azimuth_time_test['secondary'])],
            ]
        )
    weather_model_path_dict = (weather_model_dict_for_center_time_test
                               if interp_method == 'center_time'
                               else weather_model_dict_for_azimuth_time_test)
    side_effect = weather_model_path_dict[weather_model_name]
    # RAiDER needs strings for paths
    side_effect = list(map(str, side_effect))
    mocker.patch('RAiDER.processWM.prepareWeatherModel',
                 side_effect=side_effect)
    iargs = [
             '--file', str(out_path),
             '--weather-model', weather_model_name,
             '-interp', interp_method
             ]
    calcDelaysGUNW(iargs)

    # Check metadata
    model_times_dict = {'reference': ["20210723T01:00:00", "20210723T02:00:00"],
                        'secondary': ["20210711T01:00:00", "20210711T02:00:00"]}
    time_dict = {'reference': "20210723T01:50:24",
                 'secondary': "20210711T01:50:24"}
    for insar_date in ['reference', 'secondary']:
        group = f'science/grids/corrections/external/troposphere/HRRR/{insar_date}'
        with xr.open_dataset(out_path, group=group) as ds:
            center_time = time_dict[insar_date]
            model_times_used = model_times_dict[insar_date]
            for var in ['troposphereWet', 'troposphereHydrostatic']:
                assert ds[var].attrs['time_interpolation_method'] == interp_method
                assert ds[var].attrs['scene_center_time'] == center_time
                assert ds[var].attrs['model_times_used'] == model_times_used


def test_hrrr_availability_check_using_gunw_ids(mocker):
    """Hits the HRRR servers and makes sure that for certain dates they are indeed flagged as false
    """

    # All dates in 2023 are available
    gunw_id = 'S1-GUNW-A-R-106-tops-20230108_20230101-225947-00078W_00041N-PP-4be8-v3_0_0'
    assert check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation(gunw_id)

    # 2016-08-09 16:00:00 is a missing date
    gunw_id = 'S1-GUNW-A-R-106-tops-20160809_20140101-160001-00078W_00041N-PP-4be8-v3_0_0'
    assert not check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation(gunw_id)


def test_hyp3_exits_succesfully_when_hrrr_not_available(mocker):
    """This test performs the GUNW entrypoint with bucket/prefix provided and only updates the json.
    Monkey patches the upload/download to/from s3 and the actual computation.
    """
    # 2016-08-09 16:00:00 is a missing date
    mocker.patch('RAiDER.aria.prepFromGUNW.check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation',
                 side_effect=[False])
    # The gunw id should not have a hyp3 file associated with it
    # This call will still hit the HRRR s3 API as done in the previous test
    mocker.patch("RAiDER.aws.get_s3_file", side_effect=['hyp3-job-uuid-3ad24/S1-GUNW-A-R-106-tops-20160809_20140101-160001-00078W_00041N-PP-4be8-v3_0_0.nc'])
    mocker.patch('RAiDER.aria.prepFromGUNW.check_weather_model_availability')
    iargs = [
               '--bucket', 's3://foo',
               '--bucket-prefix', 'hyp3-job-uuid-3ad24',
               '--weather-model', 'HRRR',
               '-interp', 'azimuth_time_grid'
               ]
    out = calcDelaysGUNW(iargs)
    assert out is None
    # Ensure calcDelaysGUNW in raider.py ended after it saw HRRR was not available
    RAiDER.aria.prepFromGUNW.check_weather_model_availability.assert_not_called()


def test_GUNW_workflow_fails_if_a_download_fails(gunw_azimuth_test, orbit_dict_for_azimuth_time_test, mocker):
    """Makes sure for azimuth-time-grid interpolation that an error is raised if one of the files fails to
    download and does not do additional processing"""
    # The first part is the same mock up as done in test_azimuth_timing_interp_against_center_time_interp
    # Maybe better mocks could be done - but this is sufficient or simply a factory for this test given
    # This is reused so many times.

    # For prepGUNW
    side_effect = [
        # center-time
        [Path(orbit_dict_for_azimuth_time_test['reference'])],
        # azimuth-time
        [Path(orbit_dict_for_azimuth_time_test['reference'])],
    ]
    mocker.patch('eof.download.download_eofs',
                    side_effect=side_effect)

    # These outputs are not needed since the orbits are specified above
    mocker.patch('RAiDER.s1_azimuth_timing.get_slc_id_from_point_and_time',
                side_effect=[
                                # Azimuth time
                                ['reference_slc_id'],
                                # using two "dummy" ids to mimic GUNW sec granules
                                # See docstring
                                ['secondary_slc_id', 'secondary_slc_id'],
                                ])

    mocker.patch(
        'RAiDER.s1_azimuth_timing.get_orbits_from_slc_ids',
        side_effect=[
            # For azimuth time
            [Path(orbit_dict_for_azimuth_time_test['reference'])],
            [Path(orbit_dict_for_azimuth_time_test['secondary']), Path(orbit_dict_for_azimuth_time_test['secondary'])],
        ]
    )

    # These are the important parts of this test
    # Makes sure that a value error is raised if a download fails via a Runtime Error
    # There are two weather model files required for this particular mock up. First, one fails.
    mocker.patch('RAiDER.processWM.prepareWeatherModel', side_effect=[RuntimeError, 'weather_model.nc'])
    mocker.patch('RAiDER.s1_azimuth_timing.get_s1_azimuth_time_grid')
    iargs_1 = [
               '--file', str(gunw_azimuth_test),
               '--weather-model', 'HRRR',
               '-interp', 'azimuth_time_grid'
               ]

    with pytest.raises(RuntimeError):
        calcDelaysGUNW(iargs_1)
    RAiDER.s1_azimuth_timing.get_s1_azimuth_time_grid.assert_not_called()


def test_value_error_for_file_inputs_when_no_data_available(mocker):
    """See test_hyp3_exits_succesfully_when_hrrr_not_available above

    In this case if a bucket is specified rather than a file; the program exits successfully!
    """
    mocker.patch('RAiDER.aria.prepFromGUNW.check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation',
                 side_effect=[False])
    mocker.patch('RAiDER.aria.prepFromGUNW.main')
    iargs = [
             '--file', 'foo.nc',
             '--weather-model', 'HRRR',
             '-interp', 'azimuth_time_grid'
             ]

    with pytest.raises(NoWeatherModelData):
        calcDelaysGUNW(iargs)
    RAiDER.aria.prepFromGUNW.main.assert_not_called()
