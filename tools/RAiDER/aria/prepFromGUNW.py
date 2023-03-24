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
import xarray as xr
import rasterio
import yaml
import shapely.wkt
from dataclasses import dataclass

import RAiDER
from RAiDER.utilFcns import rio_open, writeArrayToRaster
from RAiDER.logger import logger
from RAiDER.models import credentials
from eof.download import download_eofs

## cube spacing in degrees for each model
DCT_POSTING = {'HRRR': 0.05, 'HRES': 0.10, 'GMAO': 0.10, 'ERA5': 0.10, 'ERA5T': 0.10}


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

        path_orb = download_eofs([dt], [sat], save_dir=orbit_dir)

        return path_orb


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
            raise ValueError(f'Something is wrong with the yaml file {example_yaml}')

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
           'time_group': {'time': GUNWObj.mid_time, 'interpolate_time': True},
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
