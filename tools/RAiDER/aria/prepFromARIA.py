# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Brett Buzzanga
# Copyright 2022, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import numpy as np
import xarray as xr
import rasterio
import yaml
import RAiDER
from RAiDER.utilFcns import rio_open, writeArrayToRaster
from RAiDER.logger import logger


## ---------------------------------------------------- Prepare Input from GUNW
def makeLatLonGrid(f:str):
    '''
    Convert the geocoded grids to lat/lon files for input to RAiDER
    '''
    group   = 'science/grids/data'
    lat_f   = os.path.join(f'NETCDF:"{f}":{group}/latitude')
    lon_f   = os.path.join(f'NETCDF:"{f}":{group}/longitude')


    ds   = xr.open_dataset(f, group='science/grids/data')

    gt   = (0, 1, 0, 0, 0, 1)
    proj = ds['crs'].crs_wkt

    lats = ds.latitude.data
    lons = ds.longitude.data

    ySize = len(lats)
    xSize = len(lons)

    ## I think this is wrong. LATS come out ordered largest at top to smallest at bottom, regardless of ascending/descending
    ## ISCE lats for HR come out smallest at top (ascending)
    ## need to check ISCE lats for descending ifg (use LA)

    LATS  = np.tile(lats, (xSize, 1)).T
    LONS  = np.tile(lons, (ySize, 1))

    dst_lat = 'lat.geo'
    dst_lon = 'lon.geo'
    writeArrayToRaster(LATS, dst_lat, 0., 'GTiff', proj, gt)
    writeArrayToRaster(LONS, dst_lon, 0., 'GTiff', proj, gt)

    logger.debug('Wrote: %s', dst_lat)
    logger.debug('Wrote: %s', dst_lon)
    return


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


## only this one opens the product; need to get lats/lons actually
def get_bbox_GUNW(f:str, buff:float=1e5):
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
    # args      = parseCMD()
    ray_trace = True if args.slant == 'ray' else False

    for f in args.files:
        # version  = xr.open_dataset(f).attrs['version'] # not used yet
        # SNWE     = get_bbox_GUNW(f)
        wavelen  = xr.open_dataset(f, group='science/radarMetaData')['wavelength'].item()
        dates    = parse_dates_GUNW(f)
        time     = parse_time_GUNW(f)
        lookdir  = parse_look_dir(f)

        makeLOSFile(f, args.los_file)
        makeLatLonGrid(f)

        cfg  = {
               'look_dir':  lookdir,
               'weather_model': args.model,
               'aoi_group' : {'lat_file': 'lat.geo', 'lon_file': 'lon.geo'},
               'date_group': {'date_list': str(dates)},
               'time_group': {'time': time},
               'los_group' : {'ray_trace': ray_trace},
               'height_group': {'height_levels': str([100, 500, 1000])}
        }

        update_yaml(cfg, f'GUNW_{dates[0]}-{dates[1]}.yaml')
