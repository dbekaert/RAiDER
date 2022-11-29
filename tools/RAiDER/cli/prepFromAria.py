# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Brett Buzzanga
# Copyright 2022, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import glob
import numpy as np
import xarray as xr
import rasterio
from RAiDER.utilFcns import rio_open, writeArrayToRaster
from RAiDER.logger import logger


def create_parser():
    """Parse command line arguments using argparse."""
    import argparse
    p = argparse.ArgumentParser(
        description='Prepare files from ARIA-tools output to use with RAiDER')

    p.add_argument(
        'files', type=str,
        help='ARIA GUNW netcdf files (accepts single file and wildcard matching)\n'
    )

    p.add_argument(
        '-m', '--model', default='HRRR', type=str,
        help='Weather model (default=HRRR)')


    p.add_argument(
        '-s', '--slant',  choices=('projection', 'ray'),
        type=str, default='projection',
        help='Delay calculation projecting the zenith to slant or along rays ')

    p.add_argument(
        '-w', '--write', action='store_true',
        help=('Optionally write the delays into the GUNW products'))

    # Line of sight
    # p.add_argument(
    #     '--incFile', '-i', type=str,
    #     help='GDAL-readable raster image file of inclination',
    #     metavar='INC', required=True)
    # p.add_argument(
    #     '--azFile', '-a', type=str,
    #     help='GDAL-readable raster image file of azimuth',
    #     metavar='AZ', required=True)
    #
    p.add_argument(
        '--los', default='los.geo', type=str,
        help='Output  ine-of-sight filename')


    # p.add_argument(
    #     '--lat_filename', '-l', default='lat.rdr', type=str, dest='lat_file',
    #     help=('Output latitude filename'))
    # p.add_argument(
    #     '--lon_filename', '-L', default='lon.rdr', type=str, dest='lon_file',
    #     help=('Output longitude filename'))

    # p.add_argument(
    #     '--format', '-t', default='.tif', type=str, dest='fmt',
    #     help='Output file format (default=tif)')
    #
    return p


def parseCMD(iargs=None):
    p    = create_parser()
    args = p.parse_args(args=iargs)
    args.argv  = iargs if iargs else os.sys.argv[1:]
    args.files = glob.glob(args.files)

    return args


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
    return ref, sec


def parse_time_GUNW(f:str):
    """ Get the center time of the secondary date from the filename """
    tt = f.split('-')[7]
    return f'{tt[:2]}:{tt[2:4]}:{tt[4:]}'


def parse_look_dir(f:str):
    look_dir = f.split('-')[3]
    return 'right' if look_dir == 'r' else 'left'


def main():
    '''
    A command-line utility to convert ARIA standard product outputs from ARIA-tools to
    RAiDER-compatible format
    '''
    args = parseCMD()

    for f in args.files:
        # version  = xr.open_dataset(f).attrs['version'] # not used yet
        # SNWE     = get_bbox_GUNW(f)
        # ref, sec = parse_dates_GUNW(f)
        # wavelen  = xr.open_dataset(f, group='science/radarMetaData')['wavelength'].item()

        # makeLOSFile(f, args.los)
        makeLatLonGrid(f)
    # makeLatLonGrid(args.incFile, args.lon_file, args.lat_file, args.fmt)
