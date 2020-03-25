#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Author: Jeremy Maurer
# Copyright 2020, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import gdal
import numpy as np
from RAiDER.utilFcns import gdal_open, writeArrayToRaster


def parse_args():
    """Parse command line arguments using argparse."""
    import argparse
    p = argparse.ArgumentParser(
        description='Prepare files from ARIA-tools output to use with RAiDER')

    # Line of sight
    p.add_argument(
        '--incFile', '-i',type=str,
        help='GDAL-readable raster image file of inclination',
        metavar='INC', required=True)
    p.add_argument(
        '--azFile', '-a',type=str,
        help='GDAL-readable raster image file of azimuth',
        metavar='AZ', required=True)

    p.add_argument(
        '--los_filename', '-f', default = 'los.rdr', type=str, dest='los_file',
        help=('Output Line-of-sight filename'))
    p.add_argument(
        '--lat_filename', '-l', default = 'lat.rdr', type=str, dest='lat_file',
        help=('Output latitude filename'))
    p.add_argument(
        '--lon_filename', '-L', default = 'lon.rdr', type=str, dest='lon_file',
        help=('Output longitude filename'))

    p.add_argument(
        '--format', '-t',default = 'ENVI', type=str, dest='fmt',
        help=('Output file format'))

    return p.parse_args(), p


def makeLatLonGrid(inFile, lonFileName, latFileName, fmt = 'ENVI'):
    '''
    Convert the geocoded grids to lat/lon files for input to RAiDER
    '''
    ds = gdal.Open(inFile, gdal.GA_ReadOnly)
    xSize = ds.RasterXSize
    ySize = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    # Create the xy grid
    xStart = gt[0]
    yStart = gt[3]
    xStep = gt[1]
    yStep = gt[-1]

    xEnd = xStart + xStep*xSize-xStep
    yEnd = yStart + yStep*ySize-yStep
    
    x = np.arange(xStart, xEnd, xStep)
    y = np.arange(yStart, yEnd, yStep)
    X,Y = np.meshgrid(x,y)
    writeArrayToRaster(X, lonFileName, 0., fmt, proj, gt)
    writeArrayToRaster(Y, latFileName, 0., fmt, proj, gt)



def makeLOSFile(incFile, azFile, fmt = 'ENVI', filename = 'los.rdr'):
    '''
    Create a line-of-sight file from ARIA-derived azimuth and inclination files
    '''
    az, az_proj, az_gt = gdal_open(azFile, returnProj = True)
    az[az==0]=np.nan
    inc = gdal_open(incFile)

    heading = 90 - az
    heading[np.isnan(heading)] = 0.

    array_shp = np.shape(az)[:2]
    dType = az.dtype

    if 'complex' in str(dType):
        dType = gdal.GDT_CFloat32
    elif 'float' in str(dType):
        dType = gdal.GDT_Float32
    else:
        dType = gdal.GDT_Byte

    # Write the data to a file
    driver = gdal.GetDriverByName(fmt)
    ds = driver.Create(filename, array_shp[1], array_shp[0],  2, dType)
    ds.SetProjection(az_proj)
    ds.SetGeoTransform(az_gt)
    b1 = ds.GetRasterBand(1)
    b1.WriteArray(inc)
    b1.SetNoDataValue(0.)
    b2 = ds.GetRasterBand(2)
    b2.WriteArray(heading)
    b2.SetNoDataValue(0.)
    ds = None
    b1 = None
    b2 = None

    return 0


def prepFromAria():
    '''
    A command-line utility to convert ARIA standard product outputs from ARIA-tools to 
    RAiDER-compatible format
    '''
    args, p = parse_args()
    makeLOSFile(args.incFile, args.azFile, args.fmt, args.los_file)
    makeLatLonGrid(args.incFile, args.lon_file, args.lat_file, args.fmt)
