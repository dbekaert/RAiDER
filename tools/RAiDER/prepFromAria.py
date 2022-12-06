# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer
# Copyright 2020, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import rasterio
from RAiDER.utilFcns import rio_open, writeArrayToRaster


def parse_args():
    """Parse command line arguments using argparse."""
    import argparse
    p = argparse.ArgumentParser(
        description='Prepare files from ARIA-tools output to use with RAiDER')

    # Line of sight
    p.add_argument(
        '--incFile', '-i', type=str,
        help='GDAL-readable raster image file of inclination',
        metavar='INC', required=True)
    p.add_argument(
        '--azFile', '-a', type=str,
        help='GDAL-readable raster image file of azimuth',
        metavar='AZ', required=True)

    p.add_argument(
        '--los_filename', '-f', default='los.rdr', type=str, dest='los_file',
        help=('Output Line-of-sight filename'))
    p.add_argument(
        '--lat_filename', '-l', default='lat.rdr', type=str, dest='lat_file',
        help=('Output latitude filename'))
    p.add_argument(
        '--lon_filename', '-L', default='lon.rdr', type=str, dest='lon_file',
        help=('Output longitude filename'))

    p.add_argument(
        '--format', '-t', default='ENVI', type=str, dest='fmt',
        help=('Output file format'))

    return p.parse_args(), p


def makeLatLonGrid(inFile, lonFileName, latFileName, fmt='ENVI'):
    '''
    Convert the geocoded grids to lat/lon files for input to RAiDER
    '''
    ds = rasterio.open(inFile)
    xSize = ds.width
    ySize = ds.height
    gt = ds.transform.to_gdal()
    proj = ds.crs

    # Create the xy grid
    xStart = gt[0]
    yStart = gt[3]
    xStep = gt[1]
    yStep = gt[-1]

    xEnd = xStart + xStep * xSize - 0.5 * xStep
    yEnd = yStart + yStep * ySize - 0.5 * yStep

    x = np.arange(xStart, xEnd, xStep)
    y = np.arange(yStart, yEnd, yStep)
    X, Y = np.meshgrid(x, y)
    writeArrayToRaster(X, lonFileName, 0., fmt, proj, gt)
    writeArrayToRaster(Y, latFileName, 0., fmt, proj, gt)


def makeLOSFile(incFile, azFile, fmt='ENVI', filename='los.rdr'):
    '''
    Create a line-of-sight file from ARIA-derived azimuth and inclination files
    '''
    az, az_prof = rio_open(azFile, returnProj=True)
    az[az == 0] = np.nan
    inc = rio_open(incFile)

    heading = 90 - az
    heading[np.isnan(heading)] = 0.

    array_shp = np.shape(az)[:2]

    # Write the data to a file
    with rasterio.open(filename, mode="w", count=2,
                       driver=fmt, width=array_shp[1],
                       height=array_shp[0], crs=az_prof.crs,
                       transform=az_prof.transform,
                       dtype=az.dtype, nodata=0.) as dst:
        dst.write(inc, 1)
        dst.write(heading, 2)

    return 0


def prepFromAria():
    '''
    A command-line utility to convert ARIA standard product outputs from ARIA-tools to
    RAiDER-compatible format
    '''
    args, p = parse_args()
    makeLOSFile(args.incFile, args.azFile, args.fmt, args.los_file)
    makeLatLonGrid(args.incFile, args.lon_file, args.lat_file, args.fmt)

    
def main():
    prepFromAria()
