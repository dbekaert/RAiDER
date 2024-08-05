import os.path as op
from dataclasses import dataclass
from datetime import datetime

from RAiDER.llreader import BoundingBox
from RAiDER.models.weatherModel import make_weather_model_filename
from RAiDER.losreader import Raytracing, build_ray
from RAiDER.utilFcns import lla2ecef
from RAiDER.cli.validators import modelName2Module

from test import *


def update_model(wm_file: str, wm_eq_type: str, wm_dir: str = "weather_files_synth"):
    """Update weather model file by the equation to test, write it to disk

    wm_eq_type is one of: [hydro, wet_linear, wet_nonlinear]
    Hydro Refractivity = k1 * (Pressure/Temp), set Pressure = Temp
    Wet Refractivity = k2 * (RelHumidty/Temp) + k3 * (RelHumidity/Temp^2)
        for linear component, set RelHumidty = Temp, k3 = 0
        for nonlinear component, set RelHumidty = Temp^2, k2 = 0
    """
    assert (
        wm_eq_type in "hydro wet_linear wet_nonlinear"
    ), "Set  wm_eq_type to hydro, wet_linear, or wet_nonlinear"
    # initialize dummy wm to calculate constant delays
    # any model will do as 1) all constants same 2) all equations same
    model = op.basename(wm_file).split("_")[0].upper().replace("-", "")
    Obj = modelName2Module(model)[1]()
    ds = xr.open_dataset(wm_file)
    t = ds["t"]
    p = ds["p"]
    e = ds["e"]
    if wm_eq_type == "hydro":
        p = t
    elif wm_eq_type == "wet_linear":
        e = t
        Obj._k3 = 0
    elif wm_eq_type == "wet_nonlinear":
        e = t**2
        Obj._k2 = 0

    Obj._t = t
    Obj._p = p
    Obj._e = e

    # make new delays and overwrite weather model dataset
    Obj._get_wet_refractivity()
    Obj._get_hydro_refractivity()

    ds["wet"] = Obj._wet_refractivity
    ds["hydro"] = Obj._hydrostatic_refractivity

    os.makedirs(wm_dir, exist_ok=True)
    dst = op.join(wm_dir, op.basename(wm_file))

    if op.exists(dst):
        os.remove(dst)

    ds.to_netcdf(dst)

    ds.close()
    del ds
    return dst


def length_of_ray(target_xyz: list, model_zs, los, max_height):
    """Build rays at xy locations

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
        ray_lengths = build_ray(model_zs, ht, xyz, LOS, max_height)[0]
        outputArrs[hh] = ray_lengths.sum(0)
    return outputArrs


@dataclass
class StudyArea(object):
    """Object with shared parameters related to the study area

    region the short name corresponding to a specific bounding box.
    Choose from:
        LA (Los Angeles, California; mid latitude)
        Fort (Fortaleza, Brazil; equator)
    """

    def __init__(self, region: str, wmName: str, path: str):
        self.region = region
        self.wmName = wmName
        self.wd = op.join(path, "synthetic_test")
        self.orb_dir = ORB_DIR

        self.setup_region()

        self.dts = self.dt.strftime("%Y_%m_%d_T%H_%M_%S")
        self.ttime = self.dt.strftime("%H:%M:%S")

        self.wmObj = modelName2Module(self.wmName.upper().replace("-", ""))[1]()

        self.hgt_lvls = np.arange(-500, 9500, 500)
        self._cube_spacing_m = 10000.0

        aoi = BoundingBox(self.SNWE)
        aoi._cube_spacing_m = self._cube_spacing_m
        aoi.add_buffer(self.wmObj.getLLRes())

        self.los = Raytracing(self.orbit, time=self.dt)

        wm_bounds = aoi.calc_buffer_ray(
            self.los.getSensorDirection(), lookDir=self.los.getLookDirection()
        )
        self.wmObj.set_latlon_bounds(wm_bounds)
        wm_fname = make_weather_model_filename(
            self.wmName, self.dt, self.wmObj._ll_bounds
        )
        self.path_wm_real = op.join(WM_DIR, wm_fname)

        self.wm_dir_synth = op.join(self.wd, "weather_files_synth")

    def setup_region(self):
        """Setup the bounding box and choose orbit file based on region name

        Possible regions are:
            LA (Los Angeles, California; midlatitude)
            Fort (Fortaleza, Brazil; equator)
        """
        # Los Angeles, CA; Descending
        if self.region == "LA":
            self.SNWE = 33, 34, -118.25, -117.25
            self.dt = datetime(2020, 1, 30, 13, 52, 45)
            self.orbit = (
                self.orb_dir
                + "/S1B_OPER_AUX_POEORB_OPOD_20210317T025713_V20200129T225942_20200131T005942.EOF"
            )

        # Fortaleza, Brazil; Ascending
        elif self.region == "Fort":
            self.SNWE = -4.0, -3.5, -38.75, -38.25
            self.dt = datetime(2019, 11, 17, 20, 51, 58)
            self.orbit = (
                self.orb_dir
                + "/S1A_OPER_AUX_POEORB_OPOD_20210315T014833_V20191116T225942_20191118T005942.EOF"
            )

        # Utqiagvik, Alaska; Descending
        elif self.region == "AK":
            self.SNWE = 70.25, 71.50, -157.75, -155.55
            self.dt = datetime(2022, 8, 29, 17, 0, 1)
            self.orbit = (
                self.orb_dir
                + "/S1A_OPER_AUX_POEORB_OPOD_20220918T081841_V20220828T225942_20220830T005942.EOF"
            )

    def make_config_dict(self):
        dct = {
            "aoi_group": {"bounding_box": list(self.SNWE)},
            "height_group": {"height_levels": self.hgt_lvls.tolist()},
            "time_group": {"time": self.ttime, "interpolate_time": "none"},
            "date_group": {"date_list": datetime.strftime(self.dt, "%Y%m%d")},
            "cube_spacing_in_m": str(self._cube_spacing_m),
            "los_group": {"ray_trace": True, "orbit_file": self.orbit},
            "weather_model": self.wmName,
            "runtime_group": {"output_directory": self.wd},
        }
        return dct


@pytest.mark.skip()
@pytest.mark.parametrize("region", "AK LA Fort".split())
def test_dl_real(tmp_path, region, mod="ERA5"):
    """Download the real weather model to overwrite

    This 'golden dataset' shouldnt be changed
    """
    with pushd(tmp_path):
        SAobj = StudyArea(region, mod, tmp_path)
        dct_cfg = SAobj.make_config_dict()
        # set the real weather model path and download only
        dct_cfg["runtime_group"]["weather_model_directory"] = op.dirname(
            SAobj.path_wm_real
        )
        dct_cfg["download_only"] = True

        cfg = update_yaml(dct_cfg)
        ## run raider to download the real weather model
        cmd = f"raider.py {cfg}"

        proc = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, universal_newlines=True
        )
        assert proc.returncode == 0, "RAiDER did not complete successfully"


@pytest.mark.parametrize("region", "AK LA Fort".split())
def test_hydrostatic_eq(tmp_path, region, mod="ERA-5"):
    """Test hydrostatic equation: Hydro Refractivity = k1 * (Pressure/Temp)

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
        significantly different = 6 decimal places
    """
    with pushd(tmp_path):
        ## setup the config files
        SAobj = StudyArea(region, mod, tmp_path)
        dct_cfg = SAobj.make_config_dict()
        dct_cfg["runtime_group"]["weather_model_directory"] = SAobj.wm_dir_synth
        dct_cfg["download_only"] = False

        ## update the weather model; t = p for hydrostatic
        path_synth = update_model(SAobj.path_wm_real, "hydro", SAobj.wm_dir_synth)

        cfg = update_yaml(dct_cfg)

        ## run raider with the synthetic model
        cmd = f"raider.py {cfg}"
        proc = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, universal_newlines=True
        )
        assert proc.returncode == 0, "RAiDER did not complete successfully"

        # get the just created synthetic delays
        wm_name = SAobj.wmName.replace("-", "")  # incase of ERA-5
        ds = xr.open_dataset(
            f'{SAobj.wd}/{wm_name}_tropo_{SAobj.dts.replace("_", "")}_ray.nc'
        )
        da = ds["hydro"]
        ds.close()
        del ds

        # now build the rays at the unbuffered wm nodes
        max_tropo_height = SAobj.wmObj._zlevels[-1] - 1
        targ_xyz = [da.x.data, da.y.data, da.z.data]
        ray_length = length_of_ray(
            targ_xyz, SAobj.wmObj._zlevels, SAobj.los, max_tropo_height
        )

        # scale by constant (units K/Pa) to match raider (m K / Pa)
        ray_data = ray_length * SAobj.wmObj._k1

        # actual raider data
        # undo scaling of ppm;  units are  meters * K/Pa
        raid_data = da.data * 1e6

        assert np.all(np.abs(ray_data) > 1)
        assert np.all(np.abs(raid_data) > 1)

        # normalize with the theoretical data and compare difference with 0
        resid = (ray_data - raid_data) / ray_data
        np.testing.assert_almost_equal(0, resid, decimal=6)

        da.close()
        del da


@pytest.mark.parametrize("region", "AK LA Fort".split())
def test_wet_eq_linear(tmp_path, region, mod="ERA-5"):
    """Test linear part of wet equation.

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

    with pushd(tmp_path):
        # create temp directory for file that is created
        dir_to_del = "tmp_dir"
        if not os.path.exists(dir_to_del):
            os.mkdir(dir_to_del)

        ## setup the config files
        SAobj = StudyArea(region, mod, dir_to_del)
        dct_cfg = SAobj.make_config_dict()
        dct_cfg["runtime_group"]["weather_model_directory"] = SAobj.wm_dir_synth
        dct_cfg["download_only"] = False

        ## update the weather model; t = e for wet1
        path_synth = update_model(SAobj.path_wm_real, "wet_linear", SAobj.wm_dir_synth)

        cfg = update_yaml(dct_cfg)

        ## run raider with the synthetic model
        cmd = f"raider.py {cfg}"
        proc = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, universal_newlines=True
        )

        assert proc.returncode == 0, "RAiDER did not complete successfully"

        # get the just created synthetic delays
        wm_name = SAobj.wmName.replace("-", "")  # incase of ERA-5
        ds = xr.open_dataset(
            f'{SAobj.wd}/{wm_name}_tropo_{SAobj.dts.replace("_", "")}_ray.nc'
        )
        da = ds["wet"]
        ds.close()
        del ds

        # now build the rays at the unbuffered wm nodes
        max_tropo_height = SAobj.wmObj._zlevels[-1] - 1
        targ_xyz = [da.x.data, da.y.data, da.z.data]
        ray_length = length_of_ray(
            targ_xyz, SAobj.wmObj._zlevels, SAobj.los, max_tropo_height
        )

        # scale by constant (units K/Pa) to match raider (m K / Pa)
        ray_data = ray_length * SAobj.wmObj._k2

        # actual raider data
        # undo scaling of ppm;  units are  meters * K/Pa
        raid_data = da.data * 1e6

        assert np.all(np.abs(ray_data) > 1)
        assert np.all(np.abs(raid_data) > 1)

        # normalize with the theoretical data and compare difference with 0
        resid = (ray_data - raid_data) / ray_data
        np.testing.assert_almost_equal(0, resid, decimal=6)

        da.close()
        del da

        # delete temp directory
        if os.path.exists(dir_to_del):
            shutil.rmtree(dir_to_del)


@pytest.mark.parametrize("region", "AK LA Fort".split())
def test_wet_eq_nonlinear(tmp_path, region, mod="ERA-5"):
    """Test the nonlinear part of the wet equation.

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
        significantly different = 6 decimal places
    """

    with pushd(tmp_path):
        # create temporary directory for files created in function
        dir_to_del = "tmp_dir"
        if not os.path.exists(dir_to_del):
            os.mkdir(dir_to_del)

        ## setup the config files
        SAobj = StudyArea(region, mod, dir_to_del)
        dct_cfg = SAobj.make_config_dict()
        dct_cfg["runtime_group"]["weather_model_directory"] = SAobj.wm_dir_synth
        dct_cfg["download_only"] = False

        ## update the weather model; t = e for wet1
        path_synth = update_model(
            SAobj.path_wm_real, "wet_nonlinear", SAobj.wm_dir_synth
        )

        cfg = update_yaml(dct_cfg)

        ## run raider with the synthetic model
        cmd = f"raider.py {cfg}"
        proc = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, universal_newlines=True
        )
        assert proc.returncode == 0, "RAiDER did not complete successfully"

        # get the just created synthetic delays
        wm_name = SAobj.wmName.replace("-", "")  # incase of ERA-5
        ds = xr.open_dataset(
            f'{SAobj.wd}/{wm_name}_tropo_{SAobj.dts.replace("_", "")}_ray.nc'
        )
        da = ds["wet"]
        ds.close()
        del ds

        # now build the rays at the unbuffered wm nodes
        max_tropo_height = SAobj.wmObj._zlevels[-1] - 1
        targ_xyz = [da.x.data, da.y.data, da.z.data]
        ray_length = length_of_ray(
            targ_xyz, SAobj.wmObj._zlevels, SAobj.los, max_tropo_height
        )
        # scale by constant (units K/Pa) to match raider (m K^2 / Pa)
        ray_data = ray_length * SAobj.wmObj._k3

        # actual raider data
        # undo scaling of ppm;  units are  meters * K^2 /Pa
        raid_data = da.data * 1e6

        assert np.all(np.abs(ray_data) > 1)
        assert np.all(np.abs(raid_data) > 1)

        # normalize with the theoretical data and compare difference with 0
        resid = (ray_data - raid_data) / ray_data
        np.testing.assert_almost_equal(0, resid, decimal=6)

        da.close()
        os.remove("./temp.yaml")
        os.remove("./error.log")
        os.remove("./debug.log")
        del da

        # delete temp directory
        if os.path.exists(dir_to_del):
            shutil.rmtree(dir_to_del)
