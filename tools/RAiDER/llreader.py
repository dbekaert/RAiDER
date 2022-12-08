# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from abc import abstractmethod
import os

import numpy as np
import pandas as pd

from pyproj import CRS

from RAiDER.dem import download_dem
from RAiDER.interpolator import interpolateDEM
from RAiDER.utilFcns import rio_extents, rio_open, rio_profile, rio_stats, get_file_and_band


class AOI(object):
    '''
    This instantiates a generic AOI class object
    '''
    def __init__(self):
        self._bounding_box = None
        self._proj = CRS.from_epsg(4326)
    
    def type(self):
        return self._type

    def bounds(self):
        return self._bounding_box


    def projection(self):
        return self._proj


    def add_buffer(self, buffer):
        '''
        Check whether an extra lat/lon buffer is needed for raytracing
        '''
        # if raytracing, add a 1-degree buffer all around
        try:
            ll_bounds = self._bounding_box.copy()
        except AttributeError:
            ll_bounds = list(self._bounding_box)
        ll_bounds[0] = np.max([ll_bounds[0] - buffer, -90])
        ll_bounds[1] = np.min([ll_bounds[1] + buffer,  90])
        ll_bounds[2] = np.max([ll_bounds[2] - buffer,-180])
        ll_bounds[3] = np.min([ll_bounds[3] + buffer, 180])
        return ll_bounds


class StationFile(AOI):
    '''Use a .csv file containing at least Lat, Lon, and optionally Hgt_m columns'''
    def __init__(self, station_file):
        AOI.__init__(self)
        self._filename = station_file
        self._bounding_box = bounds_from_csv(station_file)
        self._type = 'station_file'

    def readLL(self):
        df = pd.read_csv(self._filename).drop_duplicates(subset=["Lat", "Lon"])
        return df['Lat'].values, df['Lon'].values

    def readZ(self):
        df = pd.read_csv(self._filename)
        if 'Hgt_m' in df.columns:
            return df['Hgt_m'].values
        else:
            zvals, metadata = download_dem(self._bounding_box)
            z_bounds = get_bbox(metadata)
            z_out = interpolateDEM(zvals, z_bounds, self.readLL(), method='nearest')
            df['Hgt_m'] = z_out
            df.to_csv(self._filename, index=False)
            self.__init__(self._filename)
            return z_out


class RasterRDR(AOI):
    def __init__(self, lat_file, lon_file=None, hgt_file=None, convention='isce'):
        AOI.__init__(self)
        self._type = 'radar_rasters'
        # allow for 2-band lat/lon raster
        if (lon_file is None):
            self._file = lat_file
        else:
            self._latfile = lat_file
            self._lonfile = lon_file
            self._proj, self._bounding_box, _ = bounds_from_latlon_rasters(lat_file, lon_file)

        # keep track of the height file
        self._hgtfile = hgt_file
        self._convention = convention

    def readLL(self):
        if self._latfile is not None:
            return rio_open(self._latfile), rio_open(self._lonfile)
        elif self._file is not None:
            return rio_open(self._file)
        else:
            raise ValueError('lat/lon files are not defined')


    def readZ(self):
        if self._hgtfile is not None:
            return rio_open(self._hgtfile)
        else:
            zvals, metadata = download_dem(
                self._bounding_box,
                writeDEM = True,
                outName = os.path.join('GLO30_fullres_dem.tif'),
            )
            z_bounds = get_bbox(metadata)
            z_out    = interpolateDEM(zvals, z_bounds, self.readLL(), method='nearest')
            return z_out


class BoundingBox(AOI):
    '''Parse a bounding box AOI'''
    def __init__(self, bbox):
        AOI.__init__(self)
        self._bounding_box = bbox
        self._type = 'bounding_box'

class GeocodedFile(AOI):
    '''Parse a Geocoded file for coordinates'''
    def __init__(self, filename, is_dem=False):
        AOI.__init__(self)
        self._filename     = filename
        self.p             = rio_profile(filename)
        self._bounding_box = rio_extents(self.p)
        self._is_dem       = is_dem
        _, self._proj, self._gt = rio_stats(filename)
        self._type = 'geocoded_file'


    def readLL(self):
        # ll_bounds are SNWE
        S, N, W, E = self._bounding_box
        w, h = self.p['width'], self.p['height']
        px   = (E - W) / w
        py   = (N - S) / h
        x = np.array([W + (t * px) for t in range(w)])
        y = np.array([S + (t * py) for t in range(h)])
        X, Y = np.meshgrid(x,y)
        return Y, X # lats, lons


    def readZ(self):
        if self._is_dem:
            return rio_open(self._filename)

        else:
            zvals, metadata = download_dem(
                self._bounding_box,
                writeDEM = True,
                outName = os.path.join('GLO30_fullres_dem.tif'),
            )
            z_bounds = get_bbox(metadata)
            z_out    = interpolateDEM(zvals, z_bounds, self.readLL(), method='nearest')
            return z_out


class Geocube(AOI):
    '''Parse a georeferenced data cube'''
    def __init__(self):
        AOI.__init__(self)
        self._type = 'geocube'
        raise NotImplementedError

    def readLL(self):
        return None


def bounds_from_latlon_rasters(latfile, lonfile):
    '''
    Parse lat/lon/height inputs and return
    the appropriate outputs
    '''
    latinfo = get_file_and_band(latfile)
    loninfo = get_file_and_band(lonfile)
    lat_stats, lat_proj, _ = rio_stats(latinfo[0], band=latinfo[1])
    lon_stats, lon_proj, _ = rio_stats(loninfo[0], band=loninfo[1])

    if lat_proj != lon_proj:
        raise ValueError('Projection information for Latitude and Longitude files does not match')

    # TODO - handle dateline crossing here
    snwe = (lat_stats.min, lat_stats.max,
            lon_stats.min, lon_stats.max)

    fname = os.path.basename(latfile).split('.')[0]

    return lat_proj, snwe, fname


def bounds_from_csv(station_file):
    '''
    station_file should be a comma-delimited file with at least "Lat"
    and "Lon" columns, which should be EPSG: 4326 projection (i.e WGS84)
    '''
    stats = pd.read_csv(station_file).drop_duplicates(subset=["Lat", "Lon"])
    if 'Hgt_m' in stats.columns:
        use_csv_heights = True
    snwe = [stats['Lat'].min(), stats['Lat'].max(), stats['Lon'].min(), stats['Lon'].max()]
    return snwe


def get_bbox(p):
    lon_w = p['transform'][2]
    lat_n = p['transform'][5]
    pix_lon = p['transform'][0]
    pix_lat = p['transform'][4]
    lon_e = lon_w + p['width'] * pix_lon
    lat_s = lat_n + p['width'] * pix_lat
    return lat_s, lat_n, lon_w, lon_e
