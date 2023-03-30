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
import xarray
import rasterio

from pyproj import CRS, Transformer
from shapely.geometry import Point, Polygon
from shapely.ops import transform

from RAiDER.dem import download_dem
from RAiDER.interpolator import interpolateDEM
from RAiDER.utilFcns import rio_extents, rio_open, rio_profile, rio_stats, get_file_and_band
from RAiDER.logger import logger


class AOI(object):
    '''
    This instantiates a generic AOI class object.

    Attributes:
       _bounding_box    - S N W E bounding box
       _proj            - pyproj-compatible CRS
       _type            - Type of AOI
    '''
    def __init__(self):
        self._output_directory = os.getcwd()
        self._bounding_box = None
        self._proj = CRS.from_epsg(4326)
        self._geotransform = None


    def type(self):
        return self._type


    def bounds(self):
        return list(self._bounding_box).copy()


    def geotransform(self):
        return self._geotransform


    def projection(self):
        return self._proj


    def add_buffer(self, buffer=0.5, digits=2):
        """
        Add a fixed buffer to the AOI. Ensures cube is slighly larger than requested area.

        Args:
            buffer (float)  - decimal degrees to be added to the bounding box
            digits (int)    - number of decimal digits to include in the output

        Returns:
            None. Updates self._bounding_box

        Example:
        >>> from RAiDER.models.hrrr import HRRR
        >>> from RAiDER.llreader import BoundingBox
        >>> wm = HRRR()
        >>> aoi = BoundingBox([37, 38, -92, -91])
        >>> aoi.add_buffer(buffer = 1.5 * wm.getLLRes())
        >>> aoi.bounds()
         [36.93, 38.07, -92.07, -90.93]
        """
        S, N, W, E = self.bounds()

        S, N = np.max([S-buffer, -90]),  np.min([N+buffer, 90])
        W, E = W-buffer, E+buffer # will need to handle dateline crossings elsewhere

        if np.max([np.abs(W), np.abs(E)]) > 180:
            logger.warning('Bounds extend past +/- 180. Results may be incorrect.')

        self._bounding_box = [np.round(a, digits) for a in (S, N, W, E)]
        return


    def calc_buffer_ray(self, direction, lookDir='right', incAngle=30, maxZ=80, digits=2):
        """
        Calculate the buffer for ray tracing. This only needs to be done in the east-west
        direction due to satellite orbits, and only needs extended on the side closest to
        the sensor.

        Args:
            lookDir (str)      - Sensor look direction, can be "right" or "left"
            losAngle (float)   - Incidence angle in degrees
            maxZ (float)       - maximum integration elevation in km
        """
        direction = direction.lower()
        # for isce object
        try:
            lookDir = lookDir.name.lower()
        except AttributeError:
            lookDir = lookDir.lower()

        assert direction in 'asc desc'.split(), \
            f'Incorrection orbital direction: {direction}. Choose asc or desc.'
        assert lookDir in 'right light'.split(), \
            f'Incorrection look direction: {lookDir}. Choose right or left.'

        S, N, W, E = self.bounds()

        # use a small look angle to calculate near range
        lat_max = np.max([np.abs(S), np.abs(N)])
        near    = maxZ * np.tan(np.deg2rad(incAngle))
        buffer  = near / (np.cos(np.deg2rad(lat_max)) * 100)

        # buffer on the side nearest the sensor
        if ((lookDir == 'right') and (direction == 'asc')) or ((lookDir == 'left') and (direction == 'desc')):
            W = W - buffer
        else:
            E = E + buffer

        bounds = [np.round(a, digits) for a in (S, N, W, E)]
        if np.max([np.abs(W), np.abs(E)]) > 180:
            logger.warning('Bounds extend past +/- 180. Results may be incorrect.')
        return bounds


    def set_output_directory(self, output_directory):
        self._output_directory = output_directory
        return



class StationFile(AOI):
    '''Use a .csv file containing at least Lat, Lon, and optionally Hgt_m columns'''
    def __init__(self, station_file, demFile=None):
        super().__init__()
        self._filename = station_file
        self._demfile  = demFile
        self._bounding_box = bounds_from_csv(station_file)
        self._type = 'station_file'


    def readLL(self):
        df = pd.read_csv(self._filename).drop_duplicates(subset=["Lat", "Lon"])
        return df['Lat'].values, df['Lon'].values


    def readZ(self):
        df = pd.read_csv(self._filename).drop_duplicates(subset=["Lat", "Lon"])
        if 'Hgt_m' in df.columns:
            return df['Hgt_m'].values
        else:
            demFile = os.path.join(self._output_directory, 'GLO30_fullres_dem.tif') \
                            if self._demfile is None else self._demfile

            zvals, metadata = download_dem(
                self._bounding_box,
                writeDEM=True,
                outName=demFile,
            )
            ## select instead
            z_out0 = interpolateDEM(demFile, self.readLL())
            if np.isnan(z_out0).all():
                raise Exception('DEM interpolation failed. Check DEM bounds and station coords.')


            # the diagonal is the actual stations coordinates
            z_out = np.diag(z_out0)
            df['Hgt_m'] = z_out
            df.to_csv(self._filename, index=False)
            self.__init__(self._filename)
            return z_out


class RasterRDR(AOI):
    def __init__(self, lat_file, lon_file=None, hgt_file=None, dem_file=None, convention='isce'):
        super().__init__()
        self._type = 'radar_rasters'
        self._latfile = lat_file
        self._lonfile = lon_file

        if (self._latfile is None) and (self._lonfile is None):
            raise ValueError('You need to specify a 2-band file or two single-band files')

        if not os.path.exists(self._latfile):
            raise ValueError(f'{self._latfile} cannot be found!')

        try:
            bpg = bounds_from_latlon_rasters(lat_file, lon_file)
            self._bounding_box, self._proj, self._geotransform = bpg
        except rasterio.errors.RasterioIOError:
            raise ValueError(f'Could not open {self._latfile}')

        # keep track of the height file, dem and convention
        self._hgtfile = hgt_file
        self._demfile = dem_file
        self._convention = convention


    def readLL(self):
        # allow for 2-band lat/lon raster
        lats = rio_open(self._latfile)

        if self._lonfile is None:
            return lats
        else:
            return lats, rio_open(self._lonfile)


    def readZ(self):
        if self._hgtfile is not None and os.path.exists(self._hgtfile):
            logger.info('Using existing heights at: %s', self._hgtfile)
            return rio_open(self._hgtfile)

        else:
            demFile = os.path.join(self._output_directory, 'GLO30_fullres_dem.tif') \
                            if self._demfile is None else self._demfile

            zvals, metadata = download_dem(
                self._bounding_box,
                writeDEM=True,
                outName=demFile,
            )
            z_out = interpolateDEM(demFile, self.readLL())

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
        super().__init__()
        self._filename     = filename
        self.p             = rio_profile(filename)
        self._bounding_box = rio_extents(self.p)
        self._is_dem       = is_dem
        _, self._proj, self._geotransform = rio_stats(filename)
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
        demFile = self._filename if self._is_dem else 'GLO30_fullres_dem.tif'
        bbox    = self._bounding_box
        zvals, metadata = download_dem(bbox, writeDEM=True, outName=demFile)
        z_out = interpolateDEM(demFile, self.readLL())


        return z_out


class Geocube(AOI):
    """ Pull lat/lon/height from a georeferenced data cube """
    def __init__(self, path_cube):
        super().__init__()
        self.path  = path_cube
        self._type = 'Geocube'
        self._bounding_box = self.get_extent()
        _, self._proj, self._geotransform = rio_stats(filename)

    def get_extent(self):
        with xarray.open_dataset(self.path) as ds:
            S, N = ds.latitude.min().item(), ds.latitude.max().item()
            W, E = ds.longitude.min().item(), ds.longitude.max().item()
        return [S, N, W, E]


    ## untested
    def readLL(self):
        with xarray.open_dataset(self.path) as ds:
            lats = ds.latitutde.data()
            lons = ds.longitude.data()
        Lats, Lons = np.meshgrid(lats, lons)
        return Lats, Lons

    def readZ(self):
        with xarray.open_dataset(self.path) as ds:
            heights = ds.heights.data
        return heights


def bounds_from_latlon_rasters(latfile, lonfile):
    '''
    Parse lat/lon/height inputs and return
    the appropriate outputs
    '''
    latinfo = get_file_and_band(latfile)
    loninfo = get_file_and_band(lonfile)
    lat_stats, lat_proj, lat_gt = rio_stats(latinfo[0], band=latinfo[1])
    lon_stats, lon_proj, lon_gt = rio_stats(loninfo[0], band=loninfo[1])

    if lat_proj != lon_proj:
        raise ValueError('Projection information for Latitude and Longitude files does not match')

    if lat_gt != lon_gt:
        raise ValueError('Affine transform for Latitude and Longitude files does not match')

    # TODO - handle dateline crossing here
    snwe = (lat_stats.min, lat_stats.max,
            lon_stats.min, lon_stats.max)

    if lat_proj is None:
        logger.debug('Assuming lat/lon files are in EPSG:4326')
        lat_proj = CRS.from_epsg(4326)

    return snwe, lat_proj, lat_gt


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