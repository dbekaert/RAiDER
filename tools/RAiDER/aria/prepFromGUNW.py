# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Brett Buzzanga
# Copyright 2022, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import datetime as dt
import sys
from pathlib import Path

from RAiDER.aria.types import CalcDelaysArgs
import numpy as np
import pandas as pd
import rasterio
import shapely.wkt
import xarray as xr
import rioxarray as rio
from shapely.geometry import box

from RAiDER.logger import logger
from RAiDER.models import credentials
from RAiDER.models.customExceptions import NoWeatherModelData
from RAiDER.models.hrrr import AK_GEO, HRRR_CONUS_COVERAGE_POLYGON, check_hrrr_dataset_availability
from RAiDER.s1_azimuth_timing import get_times_for_azimuth_interpolation
from RAiDER.s1_orbits import get_orbits_from_slc_ids
from RAiDER.types import BB, LookDir
from RAiDER.utilFcns import write_yaml


# cube spacing in degrees for each model
DCT_POSTING = {'HRRR': 0.05, 'HRES': 0.10, 'GMAO': 0.10, 'ERA5': 0.10, 'ERA5T': 0.10, 'MERRA2': 0.1}


def _get_acq_time_from_gunw_id(gunw_id: str, reference_or_secondary: str) -> dt.datetime:
    # Ex: S1-GUNW-A-R-106-tops-20220115_20211222-225947-00078W_00041N-PP-4be8-v3_0_0
    if reference_or_secondary not in ['reference', 'secondary']:
        raise ValueError('Reference_or_secondary must "reference" or "secondary"')
    tokens = gunw_id.split('-')
    date_tokens = tokens[6].split('_')
    date_token = date_tokens[0] if reference_or_secondary == 'reference' else date_tokens[1]
    center_time_token = tokens[7]
    cen_acq_time = dt.datetime(
        int(date_token[:4]),
        int(date_token[4:6]),
        int(date_token[6:]),
        int(center_time_token[:2]),
        int(center_time_token[2:4]),
        int(center_time_token[4:]),
    )
    return cen_acq_time


def check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation(gunw_id: str, weather_model_name: str='hrrr') -> bool:
    """
    Determine if all the times for azimuth interpolation are available using
    Herbie. Note that not all 1 hour times are available within the said date
    range of HRRR.

    Parameters
    ----------
    gunw_id : str

    Returns:
    -------
    bool

    Example:
    check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation(S1-GUNW-A-R-106-tops-20220115_20211222-225947-00078W_00041N-PP-4be8-v3_0_0)
    should return True
    """
    ref_acq_time = _get_acq_time_from_gunw_id(gunw_id, 'reference')
    sec_acq_time = _get_acq_time_from_gunw_id(gunw_id, 'secondary')

    model_step_hours = [1 if weather_model_name == 'hrrr' else 3][0]
    ref_times_for_interp = get_times_for_azimuth_interpolation(ref_acq_time, model_step_hours)
    sec_times_for_interp = get_times_for_azimuth_interpolation(sec_acq_time, model_step_hours)
    ref_dataset_availability = list(map(check_hrrr_dataset_availability, ref_times_for_interp, [weather_model_name]*len(ref_times_for_interp)))
    sec_dataset_availability = list(map(check_hrrr_dataset_availability, sec_times_for_interp, [weather_model_name]*len(sec_times_for_interp)))

    return all(ref_dataset_availability) and all(sec_dataset_availability)


def get_slc_ids_from_gunw(gunw_path: Path, reference_or_secondary: str = 'reference') -> list[str]:
    # Example input: test/gunw_test_data/S1-GUNW-D-R-059-tops-20230320_20220418-180300-00179W_00051N-PP-c92e-v2_0_6.nc
    if reference_or_secondary not in ['reference', 'secondary']:
        raise ValueError('"reference_or_secondary" must be either "reference" or "secondary"')
    group = f'science/radarMetaData/inputSLC/{reference_or_secondary}'
    with xr.open_dataset(gunw_path, group=group) as ds:
        slc_ids = ds['L1InputGranules'].values
    return slc_ids


def get_acq_time_from_slc_id(slc_id: str) -> pd.Timestamp:
    # Example input input: test/gunw_azimuth_test_data/S1B_OPER_AUX_POEORB_OPOD_20210731T111940_V20210710T225942_20210712T005942.EOF
    ts_str = slc_id.split('_')[5]
    return pd.Timestamp(ts_str)


def check_weather_model_availability(gunw_path: Path, weather_model_name: str) -> bool:
    """
    Check weather reference and secondary dates of GUNW occur within
    weather model valid range.

    Parameters
    ----------
    gunw_path : Path
    weather_model_name : str
        Should be one of 'HRRR', 'HRES', 'ERA5', 'ERA5T', 'GMAO', 'MERRA2'.

    Returns:
    -------
    bool:
        True if both reference and secondary acquisitions are within the valid temporal and spatial ranges for the given
        weather model. We assume that reference_date > secondary_date (i.e. reference scenes are most recent)

    Raises:
    ------
    ValueError
        - If weather model is not correctly referencing the Class from RAiDER.models
    """
    ref_slc_ids = get_slc_ids_from_gunw(gunw_path, reference_or_secondary='reference')
    sec_slc_ids = get_slc_ids_from_gunw(gunw_path, reference_or_secondary='secondary')

    ref_ts = get_acq_time_from_slc_id(ref_slc_ids[0]).replace(tzinfo=dt.timezone(offset=dt.timedelta()))
    sec_ts = get_acq_time_from_slc_id(sec_slc_ids[0]).replace(tzinfo=dt.timezone(offset=dt.timedelta()))

    if weather_model_name == 'HRRR':
        try:
            weather_model_name = identify_which_hrrr(gunw_path)
        except NoWeatherModelData:
            return False

    # source: https://stackoverflow.com/a/7668273
    # Allows us to get weather models as strings
    # getattr(module, 'HRRR') will return HRRR class
    module = sys.modules['RAiDER.models']
    weather_model_names = module.__all__
    if weather_model_name not in weather_model_names:
        raise ValueError(f'The "weather_model_name" must be in {", ".join(weather_model_names)}')

    weather_model_cls = getattr(module, weather_model_name)
    weather_model = weather_model_cls()

    wm_start_date, wm_end_date = weather_model._valid_range
    if not isinstance(wm_end_date, dt.datetime):
        raise ValueError(f"the weather model's end date is not valid: {wm_end_date}")
    ref_cond = ref_ts <= wm_end_date
    sec_cond = sec_ts >= wm_start_date
    return ref_cond and sec_cond


class GUNW:
    path_gunw: Path
    wm: str  # TODO(garlic-os): probably a known weather model name
    out_dir: Path
    SNWE: BB.SNWE
    heights: list[int]
    dates: list[int]  # ints in YYYYMMDD form
    mid_time: str  # str in HH:MM:SS form
    look_dir: LookDir
    wavelength: float
    name: str
    orbit_file: ...
    spacing_m: int

    def __init__(self, path_gunw: str, wm: str, out_dir: str) -> None:
        self.path_gunw = Path(path_gunw)
        self.wm = wm
        self.out_dir = Path(out_dir)

        self.SNWE = self.get_bbox()
        self.heights = np.arange(-500, 9500, 500).tolist()
        # self.heights   = [-500, 0]
        self.dates, self.mid_time = self.get_datetimes()
        self.look_dir = self.get_look_dir()
        self.wavelength = self.get_wavelength()
        self.name = self.make_fname()
        self.orbit_file = self.get_orbit_file()
        self.spacing_m = int(DCT_POSTING[self.wm] * 1e5)
        # not implemented
        # self.spacing_m = self.calc_spacing_UTM() # probably wrong/unnecessary
        # self.lat_file, self.lon_file = self.makeLatLonGrid_native()
        # self.path_cube  = self.make_cube() # not needed

    def get_bbox(self) -> BB.SNWE:
        """Get the bounding box (SNWE) from an ARIA GUNW product."""
        with xr.open_dataset(self.path_gunw) as ds:
            poly_str = ds['productBoundingBox'].data[0].decode('utf-8')
        poly = shapely.wkt.loads(poly_str)
        W, S, E, N = poly.bounds
        return S, N, W, E

    def make_fname(self) -> str:
        """Match the ref/sec filename (SLC dates may be different around edge cases)."""
        ref, sec = self.path_gunw.name.split('-')[6].split('_')
        mid_time = self.path_gunw.name.split('-')[7]
        return f'{ref}-{sec}_{mid_time}'

    def get_datetimes(self) -> tuple[list[int], str]:
        """Get the datetimes and set the satellite for orbit."""
        ref_sec = self.get_slc_dt()
        mid_dates: list[int] = []  # dates in YYYYMMDD format
        for st, en in ref_sec:
            midpoint = st + (en - st) / 2
            mid_dates.append(int(midpoint.date().strftime('%Y%m%d')))
            mid_time = midpoint.time().strftime('%H:%M:%S')
        return mid_dates, mid_time

    def get_slc_dt(self) -> list[tuple[dt.datetime, dt.datetime]]:
        """Grab the SLC start date and time from the GUNW."""
        group = 'science/radarMetaData/inputSLC'
        lst_sten: list[tuple[dt.datetime, dt.datetime]] = []
        for key in 'reference secondary'.split():
            with xr.open_dataset(self.path_gunw, group=f'{group}/{key}') as ds:
                slcs = ds['L1InputGranules']
            nslcs = slcs.count().item()
            # single slc
            if nslcs == 1:
                slc = slcs.item()
                assert slc, f'Missing {key} SLC  metadata in GUNW: {self.f}'
                st = dt.datetime.strptime(slc.split('_')[5], '%Y%m%dT%H%M%S')
                en = dt.datetime.strptime(slc.split('_')[6], '%Y%m%dT%H%M%S')
            else:
                st, en = dt.datetime(1989, 3, 1), dt.datetime(1989, 3, 1)
                for j in range(nslcs):
                    slc = slcs.data[j]
                    if slc:
                        # get the maximum range
                        st_tmp = dt.datetime.strptime(slc.split('_')[5], '%Y%m%dT%H%M%S')
                        en_tmp = dt.datetime.strptime(slc.split('_')[6], '%Y%m%dT%H%M%S')

                        # check the second SLC is within one day of the previous
                        if st > dt.datetime(1989, 3, 1):
                            stdiff = np.abs((st_tmp - st).days)
                            endiff = np.abs((en_tmp - en).days)
                            assert stdiff < 2 and endiff < 2, 'SLCs granules are too far apart in time. Incorrect metadata'

                        st = st_tmp if st_tmp > st else st
                        en = en_tmp if en_tmp > en else en

                assert st > dt.datetime(1989, 3, 1), \
                    f'Missing {key} SLC metadata in GUNW: {self.f}'

            lst_sten.append((st, en))

        return lst_sten

    def get_look_dir(self) -> LookDir:
        look_dir = self.path_gunw.name.split('-')[3].lower()
        return 'right' if look_dir == 'r' else 'left'

    def get_wavelength(self):
        group = 'science/radarMetaData'
        with xr.open_dataset(self.path_gunw, group=group) as ds:
            wavelength = ds['wavelength'].item()
        return wavelength

    # TODO(garlic-os): sounds like this returns one thing but it returns a list?
    def get_orbit_file(self) -> list[str]:
        """Get orbit file for reference (GUNW: first & later date)."""
        orbit_dir = self.out_dir / 'orbits'
        orbit_dir.mkdir(parents=True, exist_ok=True)

        # just to get the correct satellite
        group = 'science/radarMetaData/inputSLC/reference'

        with xr.open_dataset(self.path_gunw, group=f'{group}') as ds:
            slcs = ds['L1InputGranules']
        # Convert to list of strings
        slcs_lst = [slc for slc in slcs.data.tolist() if slc]
        # Remove ".zip" from the granule ids included in this field
        slcs_lst = list(map(lambda slc: slc.replace('.zip', ''), slcs_lst))

        path_orb = get_orbits_from_slc_ids(slcs_lst)

        return [str(o) for o in path_orb]

    # ------ methods below are not used
    def get_version(self):
        with xr.open_dataset(self.path_gunw) as ds:
            version = ds.attrs['version']
        return version

    def getHeights(self):
        """Get the 4 height levels within a GUNW."""
        group = 'science/grids/imagingGeometry'
        with xr.open_dataset(self.path_gunw, group=group) as ds:
            hgts = ds.heightsMeta.data.tolist()
        return hgts

    def calc_spacing_UTM(self, posting: float = 0.01):
        """Convert desired horizontal posting in degrees to meters.

        Want to calculate delays close to native model resolution (3 km for HRR)
        """
        from RAiDER.utilFcns import WGS84_to_UTM

        group = 'science/grids/data'
        with xr.open_dataset(self.path_gunw, group=group) as ds0:
            lats = ds0.latitude.data
            lons = ds0.longitude.data

        lat0, lon0 = lats[0], lons[0]
        lat1, lon1 = lat0 + posting, lon0 + posting
        res = WGS84_to_UTM(np.array([lon0, lon1]), np.array([lat0, lat1]))
        lon_spacing_m = np.subtract(*res[2][::-1])
        lat_spacing_m = np.subtract(*res[3][::-1])
        return np.mean([lon_spacing_m, lat_spacing_m])

    def makeLatLonGrid_native(self) -> tuple[Path, Path]:
        """Make LatLonGrid at GUNW spacing (90m = 0.00083333º)."""
        group = 'science/grids/data'
        with xr.open_dataset(self.path_gunw, group=group) as ds0:
            lats = ds0.latitude.data
            lons = ds0.longitude.data

        Lat, Lon = np.meshgrid(lats, lons)

        dims = 'longitude latitude'.split()
        da_lon = xr.DataArray(Lon.T, coords=[Lon[0, :], Lat[:, 0]], dims=dims)
        da_lat = xr.DataArray(Lat.T, coords=[Lon[0, :], Lat[:, 0]], dims=dims)

        dst_lat = self.out_dir / 'latitude.geo'
        dst_lon = self.out_dir / 'longitude.geo'

        da_lat.to_netcdf(dst_lat)
        da_lon.to_netcdf(dst_lon)

        logger.debug('Wrote: %s', dst_lat)
        logger.debug('Wrote: %s', dst_lon)
        return dst_lat, dst_lon

    def make_cube(self) -> Path:
        """Make LatLonGrid at GUNW spacing (90m = 0.00083333º)."""
        group = 'science/grids/data'
        with xr.open_dataset(self.path_gunw, group=group) as ds0:
            lats0 = ds0.latitude.data
            lons0 = ds0.longitude.data

        lat_st, lat_en = np.floor(lats0.min()), np.ceil(lats0.max())
        lon_st, lon_en = np.floor(lons0.min()), np.ceil(lons0.max())

        lats = np.arange(lat_st, lat_en, DCT_POSTING[self.wm])
        lons = np.arange(lon_st, lon_en, DCT_POSTING[self.wm])

        dst_cube = self.out_dir / f'GeoCube_{self.name}.nc'
        with xr.Dataset(coords={'latitude': lats, 'longitude': lons, 'heights': self.heights}) as ds:
            ds.to_netcdf(dst_cube)

        logger.info('Wrote cube to: %s', str(dst_cube))
        return dst_cube

def main(args: CalcDelaysArgs) -> tuple[Path, float]:
    """Read parameters needed for RAiDER from ARIA Standard Products (GUNW)."""
    # Check if WEATHER MODEL API credentials hidden file exists, if not create it or raise ERROR
    credentials.check_api(args.weather_model, args.api_uid, args.api_key)

    GUNWObj = GUNW(args.file, args.weather_model, args.output_directory)

    raider_cfg = {
        'weather_model': args.weather_model,
        'look_dir': GUNWObj.look_dir,
        'aoi_group': {'bounding_box': GUNWObj.SNWE},
        'height_group': {'height_levels': GUNWObj.heights},
        'date_group': {'date_list': GUNWObj.dates},
        'time_group': {
            'time': GUNWObj.mid_time,
            # Options are 'none', 'center_time', and 'azimuth_time_grid'
            'interpolate_time': args.interpolate_time,
        },
        'los_group': {
            'ray_trace': True,
            'orbit_file': GUNWObj.orbit_file,
        },
        'runtime_group': {
            'raster_format': 'nc',
            'output_directory': args.output_directory,
            'cube_spacing_in_m': GUNWObj.spacing_m,
        },
    }

    path_cfg = Path(f'GUNW_{GUNWObj.name}.yaml')
    write_yaml(raider_cfg, path_cfg)
    return path_cfg, GUNWObj.wavelength


def identify_which_hrrr(gunw_path: Path) -> str:
    group = '/science/grids/data/'
    try:
        with xr.open_dataset(gunw_path, group=f'{group}') as ds:
            gunw_poly = box(*ds.rio.bounds())
        if HRRR_CONUS_COVERAGE_POLYGON.intersects(gunw_poly):
            weather_model_name = 'HRRR'
        elif AK_GEO.intersects(gunw_poly):
            weather_model_name = 'HRRRAK'
        else:
            raise NoWeatherModelData(
                f'GUNW {gunw_path} does not intersect with any HRRR coverage area. '
                'Please use a different weather model.'
            )
    except FileNotFoundError:
        raise NoWeatherModelData(
            f'''GUNW {gunw_path} does not exist or is not a valid HRRR file.
            Please check the file path.
            '''
        )
    return weather_model_name
