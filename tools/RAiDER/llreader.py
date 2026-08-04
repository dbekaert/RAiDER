# noqa: D100
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyproj
import xarray as xr


try:
    import pandas as pd
except ImportError:
    pd = None

from pyproj import CRS

from RAiDER.logger import logger
from RAiDER.types import BB, RIO


class AOI:
    """
    This instantiates a generic AOI class object.

    Attributes:
       _bounding_box    - S N W E bounding box
       _proj            - pyproj-compatible CRS
       _type            - Type of AOI
    """

    def __init__(self, cube_spacing_in_m: Optional[float]=None, output_directory=os.getcwd()) -> None:
        self._output_directory = output_directory
        self._bounding_box = None
        self._proj = CRS.from_epsg(4326)
        self._geotransform = None
        self._cube_spacing_m = cube_spacing_in_m
    

    def __repr__(self):
        return f'AOI: {self.__class__.__name__}({self._bounding_box}, {self._type})'

    def type(self):
        return self._type

    def bounds(self):
        return list(self._bounding_box).copy()

    def geotransform(self):
        return self._geotransform

    def projection(self):
        return self._proj

    def get_output_spacing(self, crs=4326):
        """Return the output spacing in desired units."""
        output_spacing_deg = self._output_spacing
        if not isinstance(crs, CRS):
            crs = CRS.from_epsg(crs)

        ## convert it to meters users wants a projected coordinate system
        if all(axis_info.unit_name == 'degree' for axis_info in crs.axis_info):
            output_spacing = output_spacing_deg
        else:
            output_spacing = output_spacing_deg * 1e5

        return output_spacing

    def set_output_spacing(self, ll_res=None) -> None:
        """Calculate the spacing for the output grid and weather model.

        Use the requested spacing if exists or the weather model grid itself

        Returns:
            None. Sets self._output_spacing
        """
        assert ll_res or self._cube_spacing_m, 'Must pass lat/lon resolution if _cube_spacing_m is None'

        out_spacing = self._cube_spacing_m / 1e5 if self._cube_spacing_m else ll_res

        logger.debug(f'Output cube spacing: {out_spacing} degrees')
        self._output_spacing = out_spacing

    def add_buffer(self, ll_res, digits=2) -> None:
        """
        Add a fixed buffer to the AOI, accounting for the cube spacing.

        Ensures cube is slighly larger than requested area.
        The AOI will always be in EPSG:4326
        Args:
            ll_res          - weather model lat/lon resolution
            digits          - number of decimal digits to include in the output

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
        from RAiDER.utilFcns import clip_bbox

        ## add an extra buffer around the user specified region
        S, N, W, E = self.bounds()
        buffer = 1.5 * ll_res
        S, N = np.max([S - buffer, -90]), np.min([N + buffer, 90])
        W, E = W - buffer, E + buffer  # TODO: handle dateline crossings

        ## clip the buffered region to a multiple of the spacing
        self.set_output_spacing(ll_res)
        S, N, W, E = clip_bbox([S, N, W, E], self._output_spacing)

        if np.max([np.abs(W), np.abs(E)]) > 180:
            logger.warning('Bounds extend past +/- 180. Results may be incorrect.')

        self._bounding_box = [np.round(a, digits) for a in (S, N, W, E)]


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

        assert direction in 'asc desc'.split(), f'Incorrect orbital direction: {direction}. Choose asc or desc.'
        assert lookDir in 'right left'.split(), f'Incorrect look direction: {lookDir}. Choose right or left.'

        S, N, W, E = self.bounds()

        # use a small look angle to calculate near range
        lat_max = np.max([np.abs(S), np.abs(N)])
        near = maxZ * np.tan(np.deg2rad(incAngle))
        buffer = near / (np.cos(np.deg2rad(lat_max)) * 100)

        # buffer on the side nearest the sensor
        if (lookDir == 'right' and direction == 'asc') or (lookDir == 'left' and direction == 'desc'):
            W = W - buffer
        else:
            E = E + buffer

        bounds = [np.round(a, digits) for a in (S, N, W, E)]
        if np.max([np.abs(W), np.abs(E)]) > 180:
            logger.warning('Bounds extend past +/- 180. Results may be incorrect.')
        return bounds

    def set_output_directory(self, output_directory) -> None:
        self._output_directory = output_directory

    def set_output_xygrid(self, dst_crs: Union[int, str]=4326) -> None:
        """Define the locations where the delays will be returned."""
        from RAiDER.utilFcns import transform_bbox

        try:
            out_proj = CRS.from_epsg(dst_crs.replace('EPSG:', ''))
        except AttributeError:
            try:
                out_proj = CRS.from_epsg(dst_crs)
            except pyproj.exceptions.CRSError:
                out_proj = dst_crs

        out_snwe = transform_bbox(self.bounds(), src_crs=4326, dest_crs=out_proj)
        logger.debug(f'Output SNWE: {out_snwe}')

        # Build the output grid
        out_spacing = self.get_output_spacing(out_proj)
        self.xpts = np.arange(out_snwe[2], out_snwe[3] + out_spacing, out_spacing)
        self.ypts = np.arange(out_snwe[1], out_snwe[0] - out_spacing, -out_spacing)


def _parse_crs(crs: Union[int, str, CRS]) -> CRS:
    """Parse an EPSG code (int or str), CRS string, or CRS object into a pyproj CRS."""
    if isinstance(crs, CRS):
        return crs
    try:
        return CRS.from_epsg(crs)
    except pyproj.exceptions.CRSError:
        return CRS(crs)


def _ellipsoidal_to_geometric(
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    crs: CRS,
) -> np.ndarray:
    """Convert heights from ellipsoidal to geometric (geoid-referenced) if needed.

    ERA5 geopotential heights are referenced to the geoid (mean sea level).
    GNSS-derived station heights (e.g. from UNR MAGNET in IGS20) are ellipsoidal
    heights above the WGS84 ellipsoid.  In CONUS the geoid sits ~15–35 m below
    the ellipsoid, so sampling the RAiDER cube with uncorrected GNSS heights
    places the integration surface too low by that amount, causing a ~5–9 mm
    positive ZTD bias.

    Conversion targets EPSG:9707 (WGS 84 + EGM96 height).  PROJ requires the
    EGM96 geoid grid, which ships with the ``proj-data`` conda package.  If the
    grid is absent locally the function transparently enables the PROJ CDN for a
    one-time download and then restores the previous network state.  If the
    download also fails, input heights are returned unchanged with a warning.

    Args:
        lats:    station latitudes in degrees
        lons:    station longitudes in degrees
        heights: station heights in the coordinate system described by ``crs``
        crs:     pyproj CRS describing the input height datum

    Returns:
        heights above the EGM96 geoid (~MSL / orthometric),
        or the input heights unchanged if the CRS has no 3D ellipsoidal vertical.
    """
    # A 2D CRS (e.g. EPSG:4326) carries no vertical datum — return unchanged.
    if len(crs.axis_info) < 3:
        return heights

    _TARGET = CRS.from_epsg(9707)  # WGS 84 + EGM96 height

    def _build_and_apply() -> tuple:
        """Return (transformer, converted_heights)."""
        from pyproj import Transformer
        t = Transformer.from_crs(crs, _TARGET, always_xy=True)
        _, _, h = t.transform(lons, lats, heights)
        return t, h

    try:
        t, h_geoid = _build_and_apply()

        # PROJ silently falls back to a noop when the EGM96 grid is missing.
        # Detect this by inspecting the WKT pipeline string.
        # to_proj4() returns '+proj=noop' when PROJ fell back to a no-op
        # because the EGM96 grid is absent.  A real geoid transform returns
        # None (it is too complex to express as a PROJ4 string).
        if t.to_proj4() == '+proj=noop':
            logger.info(
                'EGM96 geoid grid not found locally; attempting download from '
                'the PROJ CDN.  Install the proj-data conda package for fully '
                'offline use.'
            )
            _was_enabled = pyproj.network.is_network_enabled()
            pyproj.network.set_network_enabled(True)
            try:
                t, h_geoid = _build_and_apply()
                if t.to_proj4() == '+proj=noop':
                    logger.warning(
                        'EGM96 grid unavailable (no local data and CDN '
                        'unreachable); ellipsoidal heights used unchanged.'
                    )
                    return heights
            finally:
                pyproj.network.set_network_enabled(_was_enabled)

        logger.debug(
            'Converted ellipsoidal heights to EGM96 geoid heights; '
            'mean geoid undulation applied: %.2f m',
            float(np.nanmean(h_geoid - heights)),
        )
        return h_geoid

    except Exception as exc:
        logger.warning(
            'Ellipsoidal-to-geoid height conversion failed (%s); '
            'using input heights unchanged.  '
            'Ensure proj-data grids are installed for accurate results.',
            exc,
        )
        return heights


class StationFile(AOI):
    """Use a .csv file containing at least Lat, Lon, and optionally Hgt_m columns.

    By default heights are assumed to be already geoid-referenced (MSL), matching
    the ERA5 z-axis convention (``crs=4326``).  Pass ``crs=4979`` when the file
    contains WGS84 ellipsoidal heights (e.g. from UNR MAGNET / IGS20) so that
    they are automatically converted to geoid heights before the weather model
    cube is sampled.
    """

    def __init__(
        self,
        station_file: Union[str, Path],
        demFile: Optional[Union[str, Path]] = None,
        cube_spacing_in_m: Optional[float] = None,
        output_directory: Union[str, Path] = Path.cwd(),
        crs: Union[int, str, CRS] = 4326,
    ) -> None:
        super().__init__(cube_spacing_in_m, output_directory)
        self._filename = station_file
        self._demfile = demFile
        self._bounding_box = bounds_from_csv(station_file)
        self._type = 'station_file'
        self._crs = _parse_crs(crs)

    def update_crs(self, new_crs: Union[int, str, CRS]) -> None:
        """Update the CRS describing the height datum of the station file.

        Args:
            new_crs: EPSG code (int or str), CRS string, or pyproj CRS object.
                     ``4326`` (default) — heights are geoid-referenced (MSL).
                     ``4979`` — heights are WGS84 ellipsoidal (e.g. from GNSS/UNR).
        """
        self._crs = _parse_crs(new_crs)

    def readLL(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the station lat/lons from the csv file."""
        df = pd.read_csv(self._filename).drop_duplicates(subset=['Lat', 'Lon'])
        return df['Lat'].to_numpy(), df['Lon'].to_numpy()

    def readZ(self):
        """Read station heights, converting to geoid heights if the CRS is 3D ellipsoidal.

        When constructed with ``crs=4979`` (or any 3D CRS with an ellipsoidal
        vertical), heights are converted from WGS84 ellipsoidal to EGM96 geoid
        heights so they are consistent with the ERA5 z-axis reference.
        DEM-derived heights are already MSL and are never converted.
        """
        df = pd.read_csv(self._filename).drop_duplicates(subset=['Lat', 'Lon'])
        if 'Hgt_m' in df.columns:
            heights = df['Hgt_m'].values
            return _ellipsoidal_to_geometric(
                df['Lat'].values, df['Lon'].values, heights, self._crs
            )
        else:
            # Download the DEM (DEM heights are MSL; no conversion needed)
            from RAiDER.dem import download_dem
            from RAiDER.interpolator import interpolateDEM

            demFile = (
                os.path.join(self._output_directory, 'GLO30_fullres_dem.tif')
                if self._demfile is None
                else self._demfile
            )

            download_dem(
                self._bounding_box,
                writeDEM=True,
                dem_path=Path(demFile),
            )

            ## interpolate the DEM to the query points
            z_out0 = interpolateDEM(demFile, self.readLL())
            if np.isnan(z_out0).all():
                raise Exception('DEM interpolation failed. Check DEM bounds and station coords.')
            z_out = np.diag(z_out0)  # the diagonal is the actual stations coordinates

            # write the elevations to the file
            df['Hgt_m'] = z_out
            df.to_csv(self._filename, index=False)
            self._demfile = None
            return z_out


class RasterRDR(AOI):
    """Use a 2-band raster file containing lat/lon coordinates."""

    def __init__(self, lat_file, lon_file=None, *, hgt_file=None, dem_file=None, convention='isce', cube_spacing_in_m: Optional[float]=None, output_directory=os.getcwd()) -> None:
        super().__init__(cube_spacing_in_m, output_directory)
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
        except Exception as e:
            raise ValueError(f'Could not read lat/lon rasters: {e}')

        # keep track of the height file, dem and convention
        self._hgtfile = hgt_file
        self._demfile = dem_file
        self._convention = convention

    def readLL(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        # allow for 2-band lat/lon raster
        from RAiDER.utilFcns import rio_open
        lats, _ = rio_open(Path(self._latfile))

        if self._lonfile is None:
            return lats, None
        else:
            lons, _ = rio_open(Path(self._lonfile))
            return lats, lons

    def readZ(self) -> np.ndarray:
        """Read the heights from the raster file, or download a DEM if not present."""
        from RAiDER.utilFcns import rio_open
        if self._hgtfile is not None and os.path.exists(self._hgtfile):
            logger.info('Using existing heights at: %s', self._hgtfile)
            hgts, _ = rio_open(self._hgtfile)
            return hgts

        else:
            # Download the DEM
            from RAiDER.dem import download_dem
            from RAiDER.interpolator import interpolateDEM

            demFile = (
                os.path.join(self._output_directory, 'GLO30_fullres_dem.tif')
                if self._demfile is None
                else self._demfile
            )

            download_dem(
                self._bounding_box,
                writeDEM=True,
                dem_path=Path(demFile),
            )
            z_out = interpolateDEM(demFile, self.readLL())

            return z_out


class BoundingBox(AOI):
    """Parse a bounding box AOI."""

    def __init__(self, bbox, cube_spacing_in_m: Optional[float]=None, output_directory=os.getcwd()) -> None:
        super().__init__(cube_spacing_in_m, output_directory)
        self._bounding_box = bbox
        self._type = 'bounding_box'


class GeocodedFile(AOI):
    """Parse a Geocoded file for coordinates."""

    p: RIO.Profile
    _bounding_box: BB.SNWE
    _is_dem: bool

    def __init__(self, path: Path, is_dem=False, cube_spacing_in_m: Optional[float]=None, output_directory=os.getcwd()) -> None:
        super().__init__(cube_spacing_in_m, output_directory)

        from RAiDER.utilFcns import rio_extents, rio_profile, rio_stats

        self._filename = path
        self.p = rio_profile(path)
        self._bounding_box = rio_extents(self.p)
        self._is_dem = is_dem
        _, self._proj, self._geotransform = rio_stats(path)
        self._type = 'geocoded_file'
        try:
            self.crs = self.p['crs']
        except KeyError:
            self.crs = None

    def readLL(self) -> tuple[np.ndarray, np.ndarray]:
        # ll_bounds are SNWE
        S, N, W, E = self._bounding_box
        w, h = self.p['width'], self.p['height']
        px = (E - W) / w
        py = (N - S) / h
        x = np.array([W + (t * px) for t in range(w)])
        y = np.array([S + (t * py) for t in range(h)])
        X, Y = np.meshgrid(x, y)
        return Y, X  # lats, lons

    def readZ(self):
        """Download a DEM for the file."""
        from RAiDER.dem import download_dem
        from RAiDER.interpolator import interpolateDEM

        demFile = self._filename if self._is_dem else 'GLO30_fullres_dem.tif'
        bbox = self._bounding_box
        _, _ = download_dem(bbox, writeDEM=True, dem_path=Path(demFile))
        z_out = interpolateDEM(demFile, self.readLL())

        return z_out


class Geocube(AOI):
    """Pull lat/lon/height from a georeferenced data cube."""

    def __init__(self, path_cube, cube_spacing_in_m: Optional[float]=None, output_directory=os.getcwd()) -> None:
        from RAiDER.utilFcns import rio_stats
        super().__init__(cube_spacing_in_m, output_directory)
        self.path = path_cube
        self._type = 'Geocube'
        self._bounding_box = self.get_extent()
        _, self._proj, self._geotransform = rio_stats(path_cube)

    def get_extent(self):
        with xr.open_dataset(self.path) as ds:
            S, N = ds['latitude'].min().item(), ds['latitude'].max().item()
            W, E = ds['longitude'].min().item(), ds['longitude'].max().item()
        return [S, N, W, E]

    ## untested
    def readLL(self) -> tuple[np.ndarray, np.ndarray]:
        with xr.open_dataset(self.path) as ds:
            lats = ds['latitutde'].data()
            lons = ds['longitude'].data()
        Lats, Lons = np.meshgrid(lats, lons)
        return Lats, Lons

    def readZ(self):
        with xr.open_dataset(self.path) as ds:
            heights = ds['heights'].data
        return heights


def bounds_from_latlon_rasters(lat_filestr: str, lon_filestr: str) -> tuple[BB.SNWE, CRS, RIO.GDAL]:
    """
    Parse lat/lon/height inputs and return
    the appropriate outputs.
    """
    from RAiDER.utilFcns import get_file_and_band, rio_stats

    latinfo = get_file_and_band(lat_filestr)
    loninfo = get_file_and_band(lon_filestr)
    lat_stats, lat_proj, lat_gt = rio_stats(latinfo[0], band=latinfo[1])
    lon_stats, lon_proj, lon_gt = rio_stats(loninfo[0], band=loninfo[1])

    assert lat_proj == lon_proj, 'Projection information for Latitude and Longitude files does not match'
    assert lat_gt == lon_gt, 'Affine transform for Latitude and Longitude files does not match'

    # TODO - handle dateline crossing here
    snwe = (lat_stats.min, lat_stats.max,
            lon_stats.min, lon_stats.max)

    if lat_proj is None:
        logger.debug('Assuming lat/lon files are in EPSG:4326')
        lat_proj = CRS.from_epsg(4326)

    return snwe, lat_proj, lat_gt


def bounds_from_csv(station_file):
    """
    station_file should be a comma-delimited file with at least "Lat"
    and "Lon" columns, which should be EPSG: 4326 projection (i.e WGS84).
    """
    stats = pd.read_csv(station_file).drop_duplicates(subset=['Lat', 'Lon'])
    snwe = [stats['Lat'].min(), stats['Lat'].max(), stats['Lon'].min(), stats['Lon'].max()]
    return snwe
