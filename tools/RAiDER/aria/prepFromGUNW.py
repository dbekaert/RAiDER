# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Brett Buzzanga
# Copyright 2022, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from datetime import datetime
import numpy as np
import eof.download
import xarray as xr
import rasterio
import pandas as pd
import yaml
import shapely.wkt
from dataclasses import dataclass
import sys
from shapely.geometry import box

import RAiDER
from RAiDER.logger import logger
from RAiDER.models import credentials
from RAiDER.models.hrrr import HRRR_CONUS_COVERAGE_POLYGON, AK_GEO, check_hrrr_dataset_availability
from RAiDER.s1_azimuth_timing import get_times_for_azimuth_interpolation
from RAiDER.s1_orbits import _ensure_orbit_credentials

## cube spacing in degrees for each model
DCT_POSTING = {'HRRR': 0.05, 'HRES': 0.10, 'GMAO': 0.10, 'ERA5': 0.10, 'ERA5T': 0.10}


def _get_acq_time_from_gunw_id(gunw_id: str, reference_or_secondary: str) -> datetime:
    # Ex: S1-GUNW-A-R-106-tops-20220115_20211222-225947-00078W_00041N-PP-4be8-v3_0_0
    if reference_or_secondary not in ['reference', 'secondary']:
        raise ValueError('Reference_or_secondary must "reference" or "secondary"')
    tokens = gunw_id.split('-')
    date_tokens = tokens[6].split('_')
    date_token = date_tokens[0] if reference_or_secondary == 'reference' else date_tokens[1]
    center_time_token = tokens[7]
    cen_acq_time = datetime(int(date_token[:4]),
                            int(date_token[4:6]),
                            int(date_token[6:]),
                            int(center_time_token[:2]),
                            int(center_time_token[2:4]),
                            int(center_time_token[4:]))
    return cen_acq_time



def check_hrrr_dataset_availablity_for_s1_azimuth_time_interpolation(gunw_id: str) -> bool:
    """Determines if all the times for azimuth interpolation are available using Herbie; note that not all 1 hour times
    are available within the said date range of HRRR.

    Parameters
    ----------
    gunw_id : str

    Returns
    -------
    bool
    """
    ref_acq_time = _get_acq_time_from_gunw_id(gunw_id, 'reference')
    sec_acq_time = _get_acq_time_from_gunw_id(gunw_id, 'secondary')

    model_step_hours = 1
    ref_times_for_interp = get_times_for_azimuth_interpolation(ref_acq_time, model_step_hours)
    sec_times_for_interp = get_times_for_azimuth_interpolation(sec_acq_time, model_step_hours)
    ref_dataset_availability = list(map(check_hrrr_dataset_availability, ref_times_for_interp))
    sec_dataset_availability = list(map(check_hrrr_dataset_availability, sec_times_for_interp))

    return all(ref_dataset_availability) and all(sec_dataset_availability)


def get_slc_ids_from_gunw(gunw_path: str,
                          reference_or_secondary: str = 'reference') -> list[str]:
    if reference_or_secondary not in ['reference', 'secondary']:
        raise ValueError('"reference_or_secondary" must be either "reference" or "secondary"')
    group = f'science/radarMetaData/inputSLC/{reference_or_secondary}'
    with xr.open_dataset(gunw_path, group=group) as ds:
        slc_ids = ds['L1InputGranules'].values
    return slc_ids


def get_acq_time_from_slc_id(slc_id: str) -> pd.Timestamp:
    ts_str = slc_id.split('_')[5]
    return pd.Timestamp(ts_str)


def check_weather_model_availability(gunw_path: str,
                                     weather_model_name: str) -> bool:
    """Checks weather reference and secondary dates of GUNW occur within
    weather model valid range

    Parameters
    ----------
    gunw_path : str
    weather_model_name : str
        Should be one of 'HRRR', 'HRES', 'ERA5', 'ERA5T', 'GMAO'.
    Returns
    -------
    bool:
        True if both reference and secondary acquisitions are within the valid range. We assume that
        reference_date > secondary_date (i.e. reference scenes are most recent)

    Raises
    ------
    ValueError
        - If weather model is not correctly referencing the Class from RAiDER.models
        - HRRR was requested and it's not in the HRRR CONUS or HRRR AK coverage area
    """
    ref_slc_ids = get_slc_ids_from_gunw(gunw_path, reference_or_secondary='reference')
    sec_slc_ids = get_slc_ids_from_gunw(gunw_path, reference_or_secondary='secondary')

    ref_ts = get_acq_time_from_slc_id(ref_slc_ids[0])
    sec_ts = get_acq_time_from_slc_id(sec_slc_ids[0])

    if weather_model_name == 'HRRR':
        group = '/science/grids/data/'
        variable = 'coherence'
        with rasterio.open(f'netcdf:{gunw_path}:{group}/{variable}') as ds:
            gunw_poly = box(*ds.bounds)
        if HRRR_CONUS_COVERAGE_POLYGON.intersects(gunw_poly):
            pass
        elif AK_GEO.intersects(gunw_poly):
            weather_model_name = 'HRRRAK'
        else:
            raise ValueError('HRRR was requested but it is not available in this area')

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
    if isinstance(wm_end_date, str) and wm_end_date == 'Present':
        wm_end_date = datetime.today() - weather_model._lag_time
    elif not isinstance(wm_end_date, datetime):
        raise ValueError(f'the weather model\'s end date is not valid: {wm_end_date}')
    ref_cond = ref_ts <= wm_end_date
    sec_cond = sec_ts >= wm_start_date
    return ref_cond and sec_cond


@dataclass
class GUNW:
    path_gunw: str
    wm: str
    out_dir: str

    def __post_init__(self):
        self.SNWE      = self.get_bbox()
        self.heights   = np.arange(-500, 9500, 500).tolist()
        # self.heights   = [-500, 0]
        self.dates, self.mid_time = self.get_datetimes()

        self.look_dir   = self.get_look_dir()
        self.wavelength = self.get_wavelength()
        self.name       = self.make_fname()
        self.OrbitFile  = self.get_orbit_file()
        self.spacing_m  = int(DCT_POSTING[self.wm] * 1e5)

        ## not implemented
        # self.spacing_m = self.calc_spacing_UTM() # probably wrong/unnecessary
        # self.lat_file, self.lon_file = self.makeLatLonGrid_native()
        # self.path_cube  = self.make_cube() # not needed


    def get_bbox(self):
        """ Get the bounding box (SNWE) from an ARIA GUNW product """
        with xr.open_dataset(self.path_gunw) as ds:
            poly_str = ds['productBoundingBox'].data[0].decode('utf-8')

        poly     = shapely.wkt.loads(poly_str)
        W, S, E, N = poly.bounds

        return [S, N, W, E]


    def make_fname(self):
        """ Match the ref/sec filename (SLC dates may be different around edge cases) """
        ref, sec = os.path.basename(self.path_gunw).split('-')[6].split('_')
        mid_time = os.path.basename(self.path_gunw).split('-')[7]
        return f'{ref}-{sec}_{mid_time}'


    def get_datetimes(self):
        """ Get the datetimes and set the satellite for orbit """
        ref_sec  = self.get_slc_dt()
        middates = []
        for aq in ref_sec:
            st, en   = aq
            midpt    = st + (en-st)/2
            middates.append(int(midpt.date().strftime('%Y%m%d')))
            midtime = midpt.time().strftime('%H:%M:%S')
        return middates, midtime


    def get_slc_dt(self):
        """ Grab the SLC start date and time from the GUNW """
        group    = 'science/radarMetaData/inputSLC'
        lst_sten = []
        for i, key in enumerate('reference secondary'.split()):
            ds   = xr.open_dataset(self.path_gunw, group=f'{group}/{key}')
            slcs = ds['L1InputGranules']
            nslcs = slcs.count().item()
            # single slc
            if nslcs == 1:
                slc    = slcs.item()
                assert slc, f'Missing {key} SLC  metadata in GUNW: {self.f}'
                st = datetime.strptime(slc.split('_')[5], '%Y%m%dT%H%M%S')
                en = datetime.strptime(slc.split('_')[6], '%Y%m%dT%H%M%S')
            else:
                st, en = datetime(1989, 3, 1), datetime(1989, 3, 1)
                for j in range(nslcs):
                    slc = slcs.data[j]
                    if slc:
                        ## get the maximum range
                        st_tmp = datetime.strptime(slc.split('_')[5], '%Y%m%dT%H%M%S')
                        en_tmp = datetime.strptime(slc.split('_')[6], '%Y%m%dT%H%M%S')

                        ## check the second SLC is within one day of the previous
                        if st > datetime(1989, 3, 1):
                            stdiff = np.abs((st_tmp - st).days)
                            endiff = np.abs((en_tmp - en).days)
                            assert stdiff < 2 and endiff < 2, 'SLCs granules are too far apart in time. Incorrect metadata'


                        st = st_tmp if st_tmp > st else st
                        en = en_tmp if en_tmp > en else en

                assert st>datetime(1989, 3, 1), f'Missing {key} SLC metadata in GUNW: {self.f}'

            lst_sten.append([st, en])

        return lst_sten


    def get_look_dir(self):
        look_dir = os.path.basename(self.path_gunw).split('-')[3].lower()
        return 'right' if look_dir == 'r' else 'left'


    def get_wavelength(self):
        group ='science/radarMetaData'
        with xr.open_dataset(self.path_gunw, group=group) as ds:
            wavelength = ds['wavelength'].item()
        return wavelength


    def get_orbit_file(self):
        """ Get orbit file for reference (GUNW: first & later date)"""
        orbit_dir = os.path.join(self.out_dir, 'orbits')
        os.makedirs(orbit_dir, exist_ok=True)

        # just to get the correct satellite
        group    = 'science/radarMetaData/inputSLC/reference'

        ds   = xr.open_dataset(self.path_gunw, group=f'{group}')
        slcs = ds['L1InputGranules']
        nslcs = slcs.count().item()

        if nslcs == 1:
            slc    = slcs.item()
        else:
            for j in range(nslcs):
                slc = slcs.data[j]
                if slc:
                    break

        sat = slc.split('_')[0]
        dt  = datetime.strptime(f'{self.dates[0]}T{self.mid_time}', '%Y%m%dT%H:%M:%S')

        _ensure_orbit_credentials()
        path_orb = eof.download.download_eofs([dt], [sat], save_dir=orbit_dir)

        return [str(o) for o in path_orb]


    ## ------ methods below are not used
    def get_version(self):
        with xr.open_dataset(self.path_gunw) as ds:
            version = ds.attrs['version']
        return version


    def getHeights(self):
        """ Get the 4 height levels within a GUNW """
        group ='science/grids/imagingGeometry'
        with xr.open_dataset(self.path_gunw, group=group) as ds:
            hgts = ds.heightsMeta.data.tolist()
        return hgts


    def calc_spacing_UTM(self, posting:float=0.01):
        """ Convert desired horizontal posting in degrees to meters

        Want to calculate delays close to native model resolution (3 km for HRR)
        """
        from RAiDER.utilFcns import WGS84_to_UTM
        group = 'science/grids/data'
        with xr.open_dataset(self.path_gunw, group=group) as ds0:
            lats = ds0.latitude.data
            lons = ds0.longitude.data


        lat0, lon0 = lats[0], lons[0]
        lat1, lon1 = lat0 + posting, lon0 + posting
        res        = WGS84_to_UTM(np.array([lon0, lon1]), np.array([lat0, lat1]))
        lon_spacing_m = np.subtract(*res[2][::-1])
        lat_spacing_m = np.subtract(*res[3][::-1])
        return np.mean([lon_spacing_m, lat_spacing_m])


    def makeLatLonGrid_native(self):
        """ Make LatLonGrid at GUNW spacing (90m = 0.00083333º) """
        group = 'science/grids/data'
        with xr.open_dataset(self.path_gunw, group=group) as ds0:
            lats = ds0.latitude.data
            lons = ds0.longitude.data

        Lat, Lon  = np.meshgrid(lats, lons)

        dims   = 'longitude latitude'.split()
        da_lon = xr.DataArray(Lon.T, coords=[Lon[0, :], Lat[:, 0]], dims=dims)
        da_lat = xr.DataArray(Lat.T, coords=[Lon[0, :], Lat[:, 0]], dims=dims)

        dst_lat = os.path.join(self.out_dir, 'latitude.geo')
        dst_lon = os.path.join(self.out_dir, 'longitude.geo')

        da_lat.to_netcdf(dst_lat)
        da_lon.to_netcdf(dst_lon)

        logger.debug('Wrote: %s', dst_lat)
        logger.debug('Wrote: %s', dst_lon)
        return dst_lat, dst_lon


    def make_cube(self):
        """ Make LatLonGrid at GUNW spacing (90m = 0.00083333º) """
        group = 'science/grids/data'
        with xr.open_dataset(self.path_gunw, group=group) as ds0:
            lats0 = ds0.latitude.data
            lons0 = ds0.longitude.data

        lat_st, lat_en = np.floor(lats0.min()), np.ceil(lats0.max())
        lon_st, lon_en = np.floor(lons0.min()), np.ceil(lons0.max())

        lats = np.arange(lat_st, lat_en, DCT_POSTING[self.wmodel])
        lons = np.arange(lon_st, lon_en, DCT_POSTING[self.wmodel])

        S, N = lats.min(), lats.max()
        W, E = lons.min(), lons.max()

        ds = xr.Dataset(coords={'latitude': lats, 'longitude': lons, 'heights': self.heights})
        dst_cube = os.path.join(self.out_dir, f'GeoCube_{self.name}.nc')
        ds.to_netcdf(dst_cube)

        logger.info('Wrote cube to: %s', dst_cube)
        return dst_cube


def update_yaml(dct_cfg:dict, dst:str='GUNW.yaml'):
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

    logger.info (f'Wrote new cfg file: %s', dst)
    return dst


def main(args):
    """ Read parameters needed for RAiDER from ARIA Standard Products (GUNW) """

    # Check if WEATHER MODEL API credentials hidden file exists, if not create it or raise ERROR
    credentials.check_api(args.weather_model, args.api_uid, args.api_key)

    GUNWObj = GUNW(args.file, args.weather_model, args.output_directory)

    raider_cfg  = {
           'weather_model': args.weather_model,
           'look_dir':  GUNWObj.look_dir,
           'cube_spacing_in_m': GUNWObj.spacing_m,
           'aoi_group' : {'bounding_box': GUNWObj.SNWE},
           'height_group' : {'height_levels': GUNWObj.heights},
           'date_group': {'date_list': GUNWObj.dates},
           'time_group': {'time': GUNWObj.mid_time,
                          # Options are 'none', 'center_time', and 'azimuth_time_grid'
                          'interpolate_time': args.interpolate_time},
           'los_group' : {'ray_trace': True,
                          'orbit_file': GUNWObj.OrbitFile,
                          'wavelength': GUNWObj.wavelength,
                          },

           'runtime_group': {'raster_format': 'nc',
                             'output_directory': args.output_directory,
                             }
    }

    path_cfg = f'GUNW_{GUNWObj.name}.yaml'
    update_yaml(raider_cfg, path_cfg)
    return path_cfg, GUNWObj.wavelength
