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
import RAiDER
from RAiDER.utilFcns import rio_open, writeArrayToRaster
from RAiDER.logger import logger

""" Should be refactored into a class that takes filename as input """
## ---------------------------------------------------- Prepare Input from GUNW
def makeLatLonGrid(f:str):
# def makeLatLonGrid(f:str, reg, out_dir):
    ds0  = xr.open_dataset(f, group='science/grids/data')

    Lat, Lon  = np.meshgrid(ds0.latitude.data, ds0.longitude.data)

    da_lat = xr.DataArray(Lat.T, coords=[Lon[0, :], Lat[:, 0]], dims='lon lat'.split())
    da_lon = xr.DataArray(Lon.T, coords=[Lon[0, :], Lat[:, 0]], dims='lon lat'.split())
    # dst_lat = op.join(out_dir, f'lat_{reg}.geo')
    # dst_lon = op.join(out_dir, f'lon_{reg}.geo')

    dst_lat = f'lat.geo'
    dst_lon = f'lon.geo'
    da_lat.to_netcdf(dst_lat)
    da_lon.to_netcdf(dst_lon)
    logger.debug('Wrote: %s', dst_lat)
    logger.debug('Wrote: %s', dst_lon)
    return dst_lat, dst_lon


def getHeights(f:str):
    ds =  xr.open_dataset(f, group='science/grids/imagingGeometry')
    hgts = ds.heightsMeta.data.tolist()
    return hgts


def makeLOSFile(f:str, filename:str):
    """ Create line-of-sight file from ARIA azimuth and incidence layers """

    group   = 'science/grids/imagingGeometry'
    azFile  = os.path.join(f'NETCDF:"{f}":{group}/azimuthAngle')
    incFile = os.path.join(f'NETCDF:"{f}":{group}/incidenceAngle')

    az, az_prof = rio_open(azFile, returnProj=True)
    az          = np.stack(az)
    az[az == 0] = np.nan
    array_shp   = az.shape[1:]

    heading     = 90 - az
    heading[np.isnan(heading)] = 0.

    inc = rio_open(incFile)
    inc = np.stack(inc)

    hgt = np.arange(inc.shape[0])
    y   = np.arange(inc.shape[1])
    x   = np.arange(inc.shape[2])

    da_inc  = xr.DataArray(inc, name='incidenceAngle',
                coords={'hgt': hgt, 'x': x, 'y': y},
                dims='hgt y x'.split())

    da_head = xr.DataArray(heading, name='heading',
                coords={'hgt': hgt, 'x': x, 'y': y},
                dims='hgt y x'.split())

    ds = xr.merge([da_head, da_inc]).assign_attrs(
                        crs=str(az_prof['crs']), geo_transform=az_prof['transform'])


    dst = f'{filename}.nc'
    ds.to_netcdf(dst)
    logger.debug('Wrote: %s', dst)
    return dst


## make not get correct S1A vs S1B
def getOrbitFile(f:str, orbit_dir='./orbits'):
    from eof.download import download_eofs
    os.makedirs(orbit_dir, exist_ok=True)
    # import s1reader
    group ='science/radarMetaData/inputSLC'
    sats  = []
    for key in 'reference secondary'.split():
        ds  = xr.open_dataset(f, group=f'{group}/{key}')
        slc = ds['L1InputGranules'].item()
        sats.append(slc.split('_')[0])
        # orbit_path = s1reader.get_orbit_file_from_dir(slc, orbit_dir, auto_download=True)

    dates = parse_dates_GUNW(f)
    time  = parse_time_GUNW(f)
    dts   = [datetime.strptime(f'{dt}T{time}', '%Y%m%dT%H:%M:%S') for dt in dates]
    paths = download_eofs(dts, sats, save_dir=orbit_dir)
    return paths


## only this one opens the product; need to get lats/lons actually
def get_bbox_GUNW(f:str, buff:float=1e-5):
    """ Get the bounding box (SNWE) from an ARIA GUNW product """
    import shapely.wkt
    ds       = xr.open_dataset(f)
    poly_str = ds['productBoundingBox'].data[0].decode('utf-8')
    poly     = shapely.wkt.loads(poly_str)
    W, S, E, N = poly.bounds

    ### buffer slightly?
    W, S, E, N = W-buff, S-buff, E+buff, N+buff
    return [S, N, W, E]


def parse_dates_GUNW(f:str):
    """ Get the ref/sec date from the filename """
    sec, ref = f.split('-')[6].split('_')

    return int(sec), int(ref)


def parse_time_GUNW(f:str):
    """ Get the center time of the secondary date from the filename """
    tt = f.split('-')[7]
    return f'{tt[:2]}:{tt[2:4]}:{tt[4:]}'


def parse_look_dir(f:str):
    look_dir = f.split('-')[3].lower()
    return 'right' if look_dir == 'r' else 'left'


def update_yaml(dct_cfg, dst='GUNW.yaml'):
    """ Write a temporary yaml file with the new 'value' for 'key', preserving parms in example_yaml"""

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

    for f in args.files:
        # version  = xr.open_dataset(f).attrs['version'] # not used yet
        # SNWE     = get_bbox_GUNW(f)

        # wavelen  = xr.open_dataset(f, group='science/radarMetaData')['wavelength'].item()
        dates    = parse_dates_GUNW(f)
        time     = parse_time_GUNW(f)
        heights  = getHeights(f)
        lookdir  = parse_look_dir(f)

        # makeLOSFile(f, args.los_file)
        f_lats, f_lons = makeLatLonGrid(f)
        # orbits     = getOrbitFile(f)

        cfg  = {
               'look_dir':  lookdir,
               'weather_model': args.model,
               'aoi_group' : {'lat_file': f_lats, 'lon_file': f_lons},
                'aoi_group': {'bounding_box': '37.129123314154995 37.9307480710763 -118.44814585278701 -115.494195892019'},
               'date_group': {'date_list': str(dates)},
               'time_group': {'time': time},
               'los_group' : {'ray_trace': False},
                              # 'los_convention': args.los_convention,
                              # 'los_cube': args.los_file},
                              # 'orbit_file': orbits},
               'height_group': {'height_levels': str(heights)},
               'runtime_group': {'raster_format': 'nc'}
        }
        path_cfg = f'GUNW_{dates[0]}-{dates[1]}.yaml'
        update_yaml(cfg, path_cfg)
        return path_cfg
