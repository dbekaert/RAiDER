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
import RAiDER
from RAiDER.utilFcns import rio_open, writeArrayToRaster
from RAiDER.logger import logger

""" Should be refactored into a class that takes filename as input """

class PrepGUNW(object):
    def __init__(self, f:str, out_dir:str):
        self.path_gunw = f
        self.out_dir   = out_dir
        self.SNWE      = self.get_bbox()
        self.heights   = self.getHeights()
        self.dates     = self.get_dates()
        self.ref_time  = self.get_time()
        self.look_dir  = self.get_look_dir()
        self.wavelength = self.get_wavelength()
        self.lat_file, self.lon_file = self.makeLatLonGrid()


    def getHeights(self):
        """ Get the 4 height levels within a GUNW """
        group ='science/grids/imagingGeometry'
        with xr.open_dataset(self.path_gunw, group=group) as ds:
            hgts = ds.heightsMeta.data.tolist()
        return hgts


    def makeLatLonGrid(self):
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


    def get_bbox(self):
        """ Get the bounding box (SNWE) from an ARIA GUNW product """
        with xr.open_dataset(self.path_gunw) as ds:
            poly_str = ds['productBoundingBox'].data[0].decode('utf-8')

        poly     = shapely.wkt.loads(poly_str)
        W, S, E, N = poly.bounds
        return [S, N, W, E]


    def get_dates(self):
        """ Get the ref/sec date from the filename """
        ref, sec = self.path_gunw.split('-')[6].split('_')

        return int(ref), int(sec)


    def get_time(self):
        """ Get the center time of the secondary date from the filename """
        tt = self.path_gunw.split('-')[7]
        return f'{tt[:2]}:{tt[2:4]}:{tt[4:]}'


    def get_look_dir(self):
        look_dir = self.path_gunw.split('-')[3].lower()
        return 'right' if look_dir == 'r' else 'left'

    def get_wavelength(self):
        group ='science/radarMetaData'
        with xr.open_dataset(self.path_gunw, group=group) as ds:
            wavelength = ds['wavelength'].item()
        return wavelength


    def get_version(self):
        # not used
        with xr.open_dataset(self.path_gunw) as ds:
            version = ds.attrs['version']
        return version



## make not get correct S1A vs S1B
def getOrbitFile(f:str, out_dir):
    from eof.download import download_eofs
    orbit_dir = os.path.join(out_dir, 'orbits')
    os.makedirs(orbit_dir, exist_ok=True)
    # import s1reader
    group ='science/radarMetaData/inputSLC'
    sats  = []
    for key in 'reference secondary'.split():
        ds   = xr.open_dataset(f, group=f'{group}/{key}')
        slcs = ds['L1InputGranules']
        ## sometimes there are more than one, but the second is blank
        ## sometimes there are more than one, but the second is same except version
        ## error out if its not blank;
        # assert np.count_nonzero(slcs.data) == 1, 'Cannot handle more than 1 SLC/GUNW'
        # slc = list(filter(None, slcs))[0].item()
        slc = slcs.data[0] # temporary hack; not going to use this anyway
        sats.append(slc.split('_')[0])
        # orbit_path = s1reader.get_orbit_file_from_dir(slc, orbit_dir, auto_download=True)

    dates = get_dates_GUNW(f)
    time  = parse_time_GUNW(f)
    dts   = [datetime.strptime(f'{dt}T{time}', '%Y%m%dT%H:%M:%S') for dt in dates]
    paths = download_eofs(dts, sats, save_dir=orbit_dir)
    return paths


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
    '''
    A command-line utility to convert ARIA standard product outputs from ARIA-tools to
    RAiDER-compatible format
    '''

    GUNWObj = PrepGUNW(args.file, args.output_directory)

    cfg  = {
           'weather_model': args.model,
           'look_dir':  GUNWObj.look_dir,
           # 'aoi_group' : {'lat_file': GUNWObj.lat_file, 'lon_file': GUNWObj.lon_file},
           'aoi_group' : {'bounding_box': GUNWObj.SNWE},
           'date_group': {'date_list': str(GUNWObj.dates)},
           'height_group': {'height_levels': GUNWObj.heights},
           'time_group': {'time': GUNWObj.ref_time},
           'los_group' : {'ray_trace': True,
                          # 'orbit_file': GUNWObj.orbit_files,
                          'wavelength': GUNWObj.wavelength,
                          },

           'runtime_group': {'raster_format': 'nc',
                             'output_directory': args.output_directory,
                             }
    }

    path_cfg = f'GUNW_{GUNWObj.dates[0]}-{GUNWObj.dates[1]}.yaml'
    update_yaml(cfg, path_cfg)
    return path_cfg, GUNWObj.wavelength
