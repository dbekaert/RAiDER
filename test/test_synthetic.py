import os, os.path as op
from dataclasses import dataclass
import yaml
import subprocess
from datetime import datetime
import numpy as np
import xarray as xr

import RAiDER
from RAiDER.llreader import BoundingBox
from RAiDER.models.gmao import GMAO
from RAiDER.models.weatherModel import make_weather_model_filename
from RAiDER.losreader import Raytracing, getTopOfAtmosphere
from RAiDER.utilFcns import lla2ecef, ecef2lla
from RAiDER.cli.validators import modelName2Module


import pytest
from test import TEST_DIR


def update_yaml(dct_cfg:dict, dst:str='temp.yaml'):
    """ Write a new yaml file from a dictionary.

    Updates parameters in the default 'raider.yaml' file.
    Each key:value pair will in 'dct_cfg' will overwrite that in the default
    """

    template_file = os.path.join(
                    os.path.dirname(RAiDER.__file__), 'cli', 'raider.yaml')

    with open(template_file, 'r') as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError(f'Something is wrong with the yaml file {template_file}')

    params = {**params, **dct_cfg}

    with open(dst, 'w') as fh:
        yaml.safe_dump(params, fh,  default_flow_style=False)

    return dst


def update_model(wm_file:str, wm_eq_type:str, wm_dir:str='weather_files_synth'):
    """ Update weather model file by the equation to test, write it to disk

    wm_eq_type is one of: [hydro, wet_linear, wet_nonlinear]
    Hydro Refractivity = k1 * (Pressure/Temp), set Pressure = Temp
    Wet Refractivity = k2 * (RelHumidty/Temp) + k3 * (RelHumidity/Temp^2)
        for linear component, set RelHumidty = Temp, k3 = 0
        for nonlinear component, set RelHumidty = Temp^2, k2 = 0
    """
    assert wm_eq_type in 'hydro wet_linear wet_nonlinear', \
                    'Set  wm_eq_type to hydro, wet_linear, or wet_nonlinear'
    from RAiDER.models.gmao import GMAO
    # initialize dummy wm to calculate constant delays
    # any model will do as 1) all constants same 2) all equations same
    model = op.basename(wm_file).split('_')[0].upper().replace("-", "")
    Obj = modelName2Module(model)[1]()
    ds = xr.open_dataset(wm_file)
    t  = ds['t']
    p  = ds['p']
    e  = ds['e']
    if wm_eq_type == 'hydro':
        p = t
    elif wm_eq_type == 'wet_linear':
        e = t
        Obj._k3 = 0
    elif wm_eq_type == 'wet_nonlinear':
        e = t**2
        Obj._k2 = 0

    Obj._t  = t
    Obj._p  = p
    Obj._e  = e

    # make new delays and overwrite weather model dataset
    Obj._get_wet_refractivity()
    Obj._get_hydro_refractivity()

    ds['wet']   = Obj._wet_refractivity
    ds['hydro']   = Obj._hydrostatic_refractivity

    os.makedirs(wm_dir, exist_ok=True)
    dst = op.join(wm_dir, op.basename(wm_file))

    if op.exists(dst):
        os.remove(dst)

    ds.to_netcdf(dst)

    ds.close()
    del ds
    print ('Wrote synthetic weather model file to:', dst)
    return dst


def length_of_ray(target_xyz:list, model_zs, los):
    """ Build rays at xy locations

    Target xyz is a list of lists (xpts, ypts, hgt_levels)
    Model_zs are all the model levels over which ray is calculated
    los in los object (has the orbit info)
    """

    # Create a regular 2D grid
    xx, yy = np.meshgrid(target_xyz[0], target_xyz[1])
    hgt_lvls = target_xyz[2]

    # Where total rays are stored,
    outputArrs = np.zeros((hgt_lvls.size, target_xyz[1].size, target_xyz[0].size))

    # iterate over height levels
    for hh, ht in enumerate(hgt_lvls):

        llh = [xx, yy, np.full(yy.shape, ht)]

        xyz = np.stack(lla2ecef(llh[1], llh[0], np.full(yy.shape, ht)), axis=-1)
        LOS = los.getLookVectors(ht, llh, xyz, yy)

        cos_factor = None

        # 2d array where output is added to; one per hgt lvl
        outSubs = outputArrs[hh, ...]
        # iterate over all model levels
        for zz in range(model_zs.size-1):
            low_ht = model_zs[zz]
            high_ht = model_zs[zz + 1]

            if (high_ht <= ht) or (low_ht >= 50000):
                continue

            # If high_ht > max_tropo_height - integral only up to max tropo
            if high_ht > 50000:
                high_ht = 50000

            # If low_ht < height of point - integral only up to height of point
            if low_ht < ht:
                low_ht = ht

            # Continue only if needed - 1m troposphere does nothing
            if np.abs(high_ht - low_ht) < 1.0:
                continue

            low_xyz = getTopOfAtmosphere(xyz, LOS, low_ht, factor=cos_factor)
            high_xyz = getTopOfAtmosphere(xyz, LOS, high_ht, factor=cos_factor)

            ray_length = np.linalg.norm(high_xyz - low_xyz, axis=-1)

            outSubs   += ray_length

            cos_factor = (high_ht - low_ht) / ray_length if cos_factor is None else cos_factor

    return outputArrs


@dataclass
class StudyArea(object):
    """ Object with shared parameters related to the study area

    region the short name corresponding to a specific bounding box.
    Choose from:
        LA (Los Angeles, California; mid latitude)
        Fort (Fortaleza, Brazil; equator)
    """
    region:str
    wmName:str
    wd = op.join(TEST_DIR, 'synthetic_test')

    def __post_init__(self):
        self.setup_region()

        self.dts   = self.dt.strftime('%Y_%m_%d_T%H_%M_%S')
        self.ttime = self.dt.strftime('%H:%M:%S')

        self.wmObj = modelName2Module(self.wmName.upper().replace("-", ""))[1]()

        aoi = BoundingBox(self.SNWE)
        aoi.add_buffer(buffer = 1.5 * self.wmObj.getLLRes())

        self.los  = Raytracing(self.orbit, time=self.dt)

        wm_bounds = aoi.calc_buffer_ray(self.los.getSensorDirection(), lookDir=self.los.getLookDirection())
        self.wmObj.set_latlon_bounds(wm_bounds)
        wm_fname  = make_weather_model_filename(self.wmName, self.dt, self.wmObj._ll_bounds)
        self.path_wm_real = op.join(self.wd, 'weather_files_real', wm_fname)

        grid_spacing = 0.5
        self.cube_spacing = np.round(grid_spacing/1e-5).astype(np.float32)
        self.hgt_lvls     = np.arange(-500, 1500, 500)


    def setup_region(self):
        """ Setup the bounding box and choose orbit file based on region name

        Possible regions are:
            LA (Los Angeles, California; midlatitude)
            Fort (Fortaleza, Brazil; equator)
        """
        # Los Angeles
        if self.region == 'LA':
            self.SNWE  = 33, 34, -118.25, -117.25
            self.dt    = datetime(2020, 1, 30, 13, 52, 45)
            self.orbit = self.wd + \
                '/S1B_OPER_AUX_POEORB_OPOD_20210317T025713_V20200129T225942_20200131T005942.EOF'

        # Fortaleza, Brazil
        elif self.region == 'Fort':
            self.SNWE = -4.0, -3.5, -38.75, -38.25
            self.dt   = datetime(2019, 11, 17, 20, 51, 58)
            self.orbit = self.wd + \
                '/S1A_OPER_AUX_POEORB_OPOD_20210315T014833_V20191116T225942_20191118T005942.EOF'


    def make_config_dict(self):
        dct = {
            'aoi_group': {'bounding_box': list(self.SNWE)},
            'height_group': {'height_levels': self.hgt_lvls.tolist()},
            'time_group': {'time': self.ttime, 'interpolate_time': False},
            'date_group': {'date_list': datetime.strftime(self.dt, '%Y%m%d')},
            'cube_spacing_in_m': str(self.cube_spacing),
            'los_group': {'ray_trace': True, 'orbit_file': self.orbit},
            'weather_model': self.wmName,
            'runtime_group': {'output_directory': self.wd},
        }
        return dct


def dl_real(region='LA', mod='GMAO'):
    """ Download the real weather model to overwrite

    This 'golden dataset' shouldnt be changed
    """
    SAobj = StudyArea(region, mod)
    dct_cfg = SAobj.make_config_dict()
    # set the real weather model path and download only
    dct_cfg['runtime_group']['weather_model_directory'] = \
            op.dirname(SAobj.path_wm_real)
    dct_cfg['download_only'] = True

    cfg = update_yaml(dct_cfg)

    ## run raider to download the real weather model
    cmd  = f'raider.py {cfg}'

    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0, 'RAiDER did not complete successfully'


def test_hydrostatic_eq(region='Fort', mod='GMAO'):
    """ Test hydrostatic equation: Hydro Refractivity = k1 * (Pressure/Temp)

    The hydrostatic delay reduces to an integral along the ray path when P=T.
    However the constants k1 and scaling of 10^-6 will remain present leading
    to a result of ray-length * k1 * 10^-6. We specifically do the following:

    Compute ray length here (length_of_ray; m).
    Run raider with P=T to return delays in m * K/Pa
    Scale ray length computed here by constant (K/Pa)
    (Un)scale raider delay by parts per million term (*1e6)
    Check they are both large enough for meaningful numerical comparison (>1)
    Compute residual and normalize by theoretical ray length (calculated here)
    Ensure that normalized residual is not significantly different from 0
        significantly different = 7 decimal places
    """

    ## setup the config files
    SAobj = StudyArea(region, mod)
    dct_cfg      = SAobj.make_config_dict()
    wm_dir_synth = op.dirname(SAobj.path_wm_real).replace('real', 'synth')
    dct_cfg['runtime_group']['weather_model_directory'] = wm_dir_synth
    dct_cfg['download_only'] = False

    ## update the weather model; t = p for hydrostatic
    path_synth = update_model(SAobj.path_wm_real, 'hydro', wm_dir_synth)

    cfg = update_yaml(dct_cfg)

    ## run raider with the synthetic model
    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0, 'RAiDER did not complete successfully'

    # get the just created synthetic delays
    ds = xr.open_dataset(
        f'{SAobj.wd}/{SAobj.wmName}_tropo_{SAobj.dts.replace("_", "")}_ray.nc')
    da = ds['hydro']
    ds.close()
    del ds

    # now build the rays at the unbuffered wm nodes
    targ_xyz = [da.x.data, da.y.data, da.z.data]
    ray_length = length_of_ray(targ_xyz, SAobj.wmObj._zlevels, SAobj.los)

    # scale by constant (units K/Pa) to match raider (m K / Pa)
    ray_data  = ray_length * SAobj.wmObj._k1

    # actual raider data
    # undo scaling of ppm;  units are  meters * K/Pa
    raid_data = da.data * 1e6

    assert np.all(np.abs(ray_data) > 1)
    assert np.all(np.abs(raid_data) > 1)

    # normalize with the theoretical data and compare difference with 0
    resid     = (ray_data - raid_data) / ray_data
    np.testing.assert_almost_equal(0, resid, decimal=7)

    da.close()
    del da


def test_wet_eq_linear(region='Fort', mod='GMAO'):
    """ Test linear part of wet equation.

    Wet Refractivity = k2 * (E/T) + k3 * (E/T^2)
    E = relative humidty; T = temperature

    The wet delay reduces to an integral along the ray path when E=T and k3 = 0
    However the constants k2 and scaling of 10^-6 will remain present, leading
    to a result of ray-lengh * k2 * 10^-6. We specifically do the following:

    Computed ray length here (length_of_ray; m).
    Run raider with E=T and k3=0 to return delays in m * K/Pa
    Scale ray length computed here by constant (K/Pa)
    (Un)scale raider delay by parts per million term (*1e6)
    Check they are both large enough for meaningful numerical comparison (>1)
    Compute residual and normalize by theoretical ray length (calculated here)
    Ensure that normalized residual is not significantly different from 0
        significantly different = 7 decimal places
    """

    ## setup the config files
    SAobj = StudyArea(region, mod)
    dct_cfg      = SAobj.make_config_dict()
    wm_dir_synth = op.dirname(SAobj.path_wm_real).replace('real', 'synth')
    dct_cfg['runtime_group']['weather_model_directory'] = wm_dir_synth
    dct_cfg['download_only'] = False

    ## update the weather model; t = e for wet1
    path_synth = update_model(SAobj.path_wm_real, 'wet_linear', wm_dir_synth)

    cfg = update_yaml(dct_cfg)

    ## run raider with the synthetic model
    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0, 'RAiDER did not complete successfully'

    # get the just created synthetic delays
    ds = xr.open_dataset(
        f'{SAobj.wd}/{SAobj.wmName}_tropo_{SAobj.dts.replace("_", "")}_ray.nc')
    da = ds['wet']
    ds.close()
    del ds

    # now build the rays at the unbuffered wm nodes
    targ_xyz = [da.x.data, da.y.data, da.z.data]
    ray_length = length_of_ray(targ_xyz, SAobj.wmObj._zlevels, SAobj.los)

    # scale by constant (units K/Pa) to match raider (m K / Pa)
    ray_data  = ray_length * SAobj.wmObj._k2

    # actual raider data
    # undo scaling of ppm;  units are  meters * K/Pa
    raid_data = da.data * 1e6

    assert np.all(np.abs(ray_data) > 1)
    assert np.all(np.abs(raid_data) > 1)

    # normalize with the theoretical data and compare difference with 0
    resid     = (ray_data - raid_data) / ray_data
    np.testing.assert_almost_equal(0, resid, decimal=7)

    da.close()
    del da


def test_wet_eq_nonlinear(region='Fort', mod='GMAO'):
    """ Test the nonlinear part of the wet equation.

    Wet Refractivity = k2 * (E/T) + k3 * (E/T^2)
    E = relative humidty; T = temperature

    The wet delay reduces to an integral along the ray path when E=T^2 and k2=0
    However the constants k3 and scaling of 10^-6 will remain present leading
    to a result of ray-lengh * k3 * 10^-6. We specifically do the following:

    Computed ray length here (length_of_ray; m), and run raider with P=T (m * K^2/Pa)
    ray length computed here is then scaled by constant K^2/Pa
    delay from raider is unscaled from parts per million (*1e6)
    We check they are both large enough for meaningful numerical comparison (>1)
    We then compute residual and normalize it with theoretical ray length
    We ensure that normalized residual is not significantly different from 0
        significantly different = 7 decimal places
    """

    ## setup the config files
    SAobj = StudyArea(region, mod)
    dct_cfg      = SAobj.make_config_dict()
    wm_dir_synth = op.dirname(SAobj.path_wm_real).replace('real', 'synth')
    dct_cfg['runtime_group']['weather_model_directory'] = wm_dir_synth
    dct_cfg['download_only'] = False

    ## update the weather model; t = e for wet1
    path_synth = update_model(SAobj.path_wm_real, 'wet_nonlinear', wm_dir_synth)

    cfg = update_yaml(dct_cfg)

    ## run raider with the synthetic model
    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0, 'RAiDER did not complete successfully'

    # get the just created synthetic delays
    ds = xr.open_dataset(
        f'{SAobj.wd}/{SAobj.wmName}_tropo_{SAobj.dts.replace("_", "")}_ray.nc')
    da = ds['wet']
    ds.close()
    del ds

    # now build the rays at the unbuffered wm nodes
    targ_xyz = [da.x.data, da.y.data, da.z.data]
    ray_length = length_of_ray(targ_xyz, SAobj.wmObj._zlevels, SAobj.los)
    # scale by constant (units K/Pa) to match raider (m K^2 / Pa)
    ray_data  = ray_length * SAobj.wmObj._k3

    # actual raider data
    # undo scaling of ppm;  units are  meters * K^2 /Pa
    raid_data = da.data * 1e6

    assert np.all(np.abs(ray_data) > 1)
    assert np.all(np.abs(raid_data) > 1)

    # normalize with the theoretical data and compare difference with 0
    resid     = (ray_data - raid_data) / ray_data
    np.testing.assert_almost_equal(0, resid, decimal=7)

    da.close()
    os.remove('./temp.yaml')
    os.remove('./error.log')
    os.remove('./debug.log')
    del da
