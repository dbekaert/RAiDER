# noqa: D100
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import warnings
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

    def __init__(self, cube_spacing_in_m: Optional[float] = None, output_directory=os.getcwd()) -> None:
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

        # convert it to meters users wants a projected coordinate system
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

        # add an extra buffer around the user specified region
        S, N, W, E = self.bounds()
        buffer = 1.5 * ll_res
        S, N = np.max([S - buffer, -90]), np.min([N + buffer, 90])
        W, E = W - buffer, E + buffer  # TODO: handle dateline crossings

        # clip the buffered region to a multiple of the spacing
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

    def set_output_xygrid(self, dst_crs: Union[int, str] = 4326) -> None:
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


_EGM2008_CRS = CRS.from_epsg(9518)  # WGS 84 + EGM2008 height (geoid-referenced)
_WGS84_3D_CRS = CRS.from_epsg(4979)  # WGS 84 3D geographic (ellipsoidal height)


def _is_geoid_crs(crs: CRS) -> bool:
    """Whether a CRS's heights are geoid-referenced rather than ellipsoidal.

    Counting axes cannot answer this: EPSG:9518 (geoid) and EPSG:4979
    (ellipsoidal) both have three. What separates them is that a geoid CRS is
    compound -- a horizontal CRS paired with a separate vertical datum --
    while a 3D geographic CRS carries an ellipsoidal height axis directly.
    """
    return crs == _EGM2008_CRS or bool(crs.sub_crs_list)


def _parse_crs(crs: Optional[Union[int, str, CRS]]) -> CRS:
    """Parse an EPSG code (int or str), CRS string, or CRS object into a pyproj CRS.

    Always returns a CRS with a real vertical axis, so it can be handed
    straight to PROJ as the source or target of a height transform:

    * None -- a blank value in a run config, which is how template.yaml leaves
      every other key -- means "unspecified" and falls back to WGS84 ellipsoidal.
    * A 2D CRS (e.g. the documented ``crs=4326``) carries no vertical axis at
      all. RAiDER has always read that as "these heights are MSL", so it is
      normalised to EPSG:9518, which says the same thing in a form PROJ can
      actually transform from. Without this, a 2D CRS reaching a transform is
      silently untransformable and the conversion is skipped.
    """
    if crs is None:
        return _WGS84_3D_CRS
    if isinstance(crs, CRS):
        parsed = crs
    else:
        try:
            parsed = CRS.from_epsg(crs)
        except pyproj.exceptions.CRSError:
            parsed = CRS(crs)

    if len(parsed.axis_info) < 3:
        return _EGM2008_CRS
    return parsed


HGT_DATUM_COLUMN = 'Hgt_datum'
_ELLIPSOIDAL_LABEL = 'ellipsoidal'
_GEOID_LABEL = 'geoid'


def _source_is_geoid(df: 'pd.DataFrame', fallback_crs: CRS) -> bool:
    """Determine whether a station file's Hgt_m values are geoid-referenced.

    Prefers the HGT_DATUM_COLUMN column if present -- it's written
    authoritatively by StationFile.readZ() itself whenever it writes
    DEM-derived heights, so it survives round-trips through the file
    regardless of what crs= a later caller happens to pass. Falls back to
    fallback_crs (the constructor's crs= argument) for files that predate
    this column.
    """
    if HGT_DATUM_COLUMN in df.columns:
        labels = df[HGT_DATUM_COLUMN].unique()
        if len(labels) != 1:
            raise ValueError(
                f'{HGT_DATUM_COLUMN} column has inconsistent values {list(labels)}; '
                'expected a single datum for the whole file.'
            )
        label = labels[0]
        if label == _GEOID_LABEL:
            return True
        if label == _ELLIPSOIDAL_LABEL:
            return False
        raise ValueError(
            f'Unrecognized {HGT_DATUM_COLUMN} value {label!r}; expected {_ELLIPSOIDAL_LABEL!r} or {_GEOID_LABEL!r}.'
        )
    return _is_geoid_crs(fallback_crs)


def _warn_conversion_unavailable(reason: str) -> None:
    """Warn loudly that a height-datum conversion did not happen.

    Silent failure here is exactly the bug this conversion machinery exists
    to prevent: heights that should have been converted between ellipsoidal
    and geoid are returned unchanged instead. Logs a warning (for anyone
    watching logs) and also raises a plain UserWarning (visible on stderr by
    default even if logging isn't being watched), naming the consequence
    rather than just the failure.
    """
    msg = (
        f'Height datum conversion unavailable ({reason}); heights used '
        'unchanged. In CONUS the geoid sits up to ~35 m from the WGS84 '
        'ellipsoid, so skipping this correction translates to a several-mm '
        'bias in the computed tropospheric delay. Install the proj-data '
        'conda package for fully offline use.'
    )
    logger.warning(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)


def _convert_height_datum(
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    src_crs: CRS,
    dst_crs: CRS,
) -> np.ndarray:
    """Transform heights from src_crs to dst_crs via PROJ.

    PROJ requires the EGM2008 geoid grid for any EGM2008 <-> ellipsoidal
    conversion, which ships with the ``proj-data`` conda package.  If the
    grid is absent locally the function transparently enables the PROJ CDN
    for a one-time download and then restores the previous network state.
    If the download also fails, input heights are returned unchanged with a
    warning.
    """
    # Same datum in and out: nothing to do. This must short-circuit rather
    # than fall through to PROJ, because an identity pipeline also reports
    # '+proj=noop' -- the very string used below to detect a missing EGM2008
    # grid -- and would otherwise raise a spurious "conversion unavailable".
    if src_crs == dst_crs:
        return heights

    def _build_and_apply() -> tuple:
        """Return (transformer, converted_heights)."""
        from pyproj import Transformer

        t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        _, _, h = t.transform(lons, lats, heights)
        return t, h

    def _is_noop(t) -> bool:
        """Whether PROJ fell back to a no-op because the EGM2008 grid is missing.

        to_proj4() is a probe, not the conversion -- if it raises, that says
        nothing about whether the transform above succeeded, so treat it as
        "not a no-op" rather than letting it discard a good result.
        """
        try:
            return t.to_proj4() == '+proj=noop'
        except Exception:
            return False

    try:
        t, h_out = _build_and_apply()

        # PROJ silently falls back to a noop when the EGM2008 grid is missing.
        # to_proj4() returns '+proj=noop' when PROJ fell back to a no-op
        # because the EGM2008 grid is absent.  A real geoid transform returns
        # None (it is too complex to express as a PROJ4 string).
        if _is_noop(t):
            logger.info(
                'EGM2008 geoid grid not found locally; attempting download from '
                'the PROJ CDN.  Install the proj-data conda package for fully '
                'offline use.'
            )
            _was_enabled = pyproj.network.is_network_enabled()
            pyproj.network.set_network_enabled(True)
            try:
                t, h_out = _build_and_apply()
                if _is_noop(t):
                    _warn_conversion_unavailable('EGM2008 grid not found locally and the PROJ CDN is unreachable')
                    return heights
            finally:
                pyproj.network.set_network_enabled(_was_enabled)

    except Warning:
        # _warn_conversion_unavailable() raises when the caller configured
        # warnings as errors (-W error, or pytest's filterwarnings=error).
        # That's an explicit request for strictness, so let it propagate --
        # catching it below would re-warn with this message as its own reason
        # (a nested duplicate) and then silently return unconverted heights,
        # which is exactly what the loud warning exists to prevent.
        raise

    except Exception as exc:
        _warn_conversion_unavailable(str(exc))
        return heights

    # Outside the try: a failure in the logging below must not discard a
    # conversion that already succeeded. np.nanmean warns (and raises, under
    # warnings-as-errors) on an all-NaN difference, which is routine for DEM
    # tiles with nodata gaps.
    try:
        offset = float(np.nanmean(h_out - heights))
    except (ValueError, RuntimeWarning):
        offset = float('nan')
    logger.debug(
        'Converted heights from %s to %s; mean offset applied: %.2f m',
        src_crs.name,
        dst_crs.name,
        offset,
    )
    return h_out


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

    Conversion targets EPSG:9518 (WGS 84 + EGM2008 height). See
    _convert_height_datum for the PROJ/network fallback behavior.

    Args:
        lats:    station latitudes in degrees
        lons:    station longitudes in degrees
        heights: station heights in the coordinate system described by ``crs``
        crs:     pyproj CRS describing the input height datum

    Returns:
        heights above the EGM2008 geoid (~MSL / orthometric). Heights that are
        already geoid-referenced are returned unchanged, via the identity
        short-circuit in _convert_height_datum.
    """
    return _convert_height_datum(lats, lons, heights, _parse_crs(crs), _EGM2008_CRS)


def _geometric_to_ellipsoidal(
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    crs: CRS = CRS.from_epsg(4979),
) -> np.ndarray:
    """Convert geoid-referenced (EGM2008) heights to WGS84 ellipsoidal heights.

    Inverse of _ellipsoidal_to_geometric.  Used when a caller needs true
    ellipsoidal heights (e.g. losreader's ECEF/LOS geometry, which requires
    height above the WGS84 ellipsoid) but the heights on hand are
    geoid-referenced -- the convention used everywhere else in RAiDER to
    match the ERA5 z-axis.

    Args:
        lats:    latitudes in degrees
        lons:    longitudes in degrees
        heights: heights above the EGM2008 geoid
        crs:     target ellipsoidal CRS (default WGS84, EPSG:4979)

    Returns:
        heights above the ellipsoid described by ``crs``.
    """
    return _convert_height_datum(lats, lons, heights, _EGM2008_CRS, crs)


class StationFile(AOI):
    """Use a .csv file containing at least Lat, Lon, and optionally Hgt_m columns.

    A ``Hgt_m`` column is assumed to be WGS84 ellipsoidal by default
    (``crs=4979``), matching GNSS station positions such as UNR MAGNET /
    IGS20. Pass ``crs=4326`` when the file holds geoid-referenced (MSL)
    heights instead.

    Heights filled in from a DEM are a separate case: GLO-30 is distributed
    against the geoid and RAiDER keeps it that way (see dem.py), so those are
    written back labelled ``geoid`` regardless of ``crs``.

    Either way, readZ() returns ellipsoidal heights by default and converts
    only when asked (``geoid_heights=True``), which is what sampling the
    weather model cube needs -- its z-axis is geoid-referenced, per ERA5.
    """

    def __init__(
        self,
        station_file: Union[str, Path],
        demFile: Optional[Union[str, Path]] = None,
        cube_spacing_in_m: Optional[float] = None,
        output_directory: Union[str, Path] = Path.cwd(),
        crs: Union[int, str, CRS] = 4979,
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
                     ``4979`` (default) — heights are WGS84 ellipsoidal (e.g. from GNSS/UNR).
                     ``4326`` — heights are geoid-referenced (MSL).
        """
        self._crs = _parse_crs(new_crs)

    def readLL(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the station lat/lons from the csv file."""
        df = pd.read_csv(self._filename).drop_duplicates(subset=['Lat', 'Lon'])
        return df['Lat'].to_numpy(), df['Lon'].to_numpy()

    def readZ(self, geoid_heights: bool = False):
        """Read station heights.

        By default (``geoid_heights=False``) returns WGS84 ellipsoidal
        heights -- the native datum for GNSS station positions.

        Pass ``geoid_heights=True`` to instead get geoid-referenced (~MSL)
        heights, needed only when sampling the weather-model cube (its
        z-axis matches the ERA5 convention, geoid-referenced).

        A ``Hgt_m`` column's datum is read from the ``Hgt_datum`` column if
        present (authoritative -- written by this method whenever it writes
        DEM-derived heights, so it can't go stale the way ``self._crs``
        could), otherwise from ``self._crs`` (default WGS84 ellipsoidal,
        ``crs=4979``) for files written before that column existed.  If the
        file has no ``Hgt_m`` column, heights are DEM-derived and therefore
        geoid-referenced whatever ``self._crs`` says; they are written back
        to the file with that datum recorded, and ``self._crs`` updated to
        match, so a later call is never misled by a datum that was declared
        for a ``Hgt_m`` column which, at that point, didn't even exist yet.
        """
        df = pd.read_csv(self._filename).drop_duplicates(subset=['Lat', 'Lon'])
        if 'Hgt_m' in df.columns:
            heights = df['Hgt_m'].values
            lats, lons = df['Lat'].values, df['Lon'].values
            source_is_geoid = _source_is_geoid(df, self._crs)
        else:
            # Download the DEM. GLO-30 is geoid-referenced and dem.py keeps
            # it that way (dst_ellipsoidal_height=False), so these heights
            # are geoid regardless of what crs= this object was given.
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

            # interpolate the DEM to the query points
            z_out0 = interpolateDEM(demFile, self.readLL())
            if np.isnan(z_out0).all():
                raise Exception('DEM interpolation failed. Check DEM bounds and station coords.')
            heights = np.diag(z_out0)  # the diagonal is the actual stations coordinates
            lats, lons = self.readLL()

            # write the elevations to the file and record the datum so a
            # later readZ() call -- on this object or a fresh one -- knows
            # what it's looking at, whatever crs= it happens to be given.
            df['Hgt_m'] = heights
            df[HGT_DATUM_COLUMN] = _GEOID_LABEL
            df.to_csv(self._filename, index=False)
            self._demfile = None
            self._crs = _EGM2008_CRS
            source_is_geoid = True

        if geoid_heights == source_is_geoid:
            return heights
        if geoid_heights:
            # _parse_crs guarantees self._crs has a vertical axis, so this is
            # always transformable. When the Hgt_datum column outranks
            # self._crs and says these are ellipsoidal, a self._crs that
            # disagrees would make src == dst and no-op, so trust the column.
            src_crs = _WGS84_3D_CRS if _is_geoid_crs(self._crs) else self._crs
            return _ellipsoidal_to_geometric(lats, lons, heights, src_crs)
        return _geometric_to_ellipsoidal(lats, lons, heights)


class RasterRDR(AOI):
    """Use a 2-band raster file containing lat/lon coordinates."""

    def __init__(
        self,
        lat_file,
        lon_file=None,
        *,
        hgt_file=None,
        dem_file=None,
        convention='isce',
        cube_spacing_in_m: Optional[float] = None,
        output_directory=os.getcwd(),
    ) -> None:
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

    def readZ(self, geoid_heights: bool = False) -> np.ndarray:
        """Read the heights from the raster file, or download a DEM if not present.

        The two sources have different datums. An existing hgt_file follows
        the ISCE convention -- heights above the WGS84 ellipsoid, matching
        RasterRDR's own lat/lon geometry rasters. A downloaded DEM is
        geoid-referenced, since GLO-30 is distributed that way and dem.py
        keeps it so (dst_ellipsoidal_height=False).

        By default (geoid_heights=False) returns WGS84 ellipsoidal heights;
        pass geoid_heights=True for geoid-referenced (~MSL) heights, which is
        what sampling the weather-model cube needs. Whichever is asked for,
        only the source that disagrees with it is converted.
        """
        from RAiDER.utilFcns import rio_open

        if self._hgtfile is not None and os.path.exists(self._hgtfile):
            logger.info('Using existing heights at: %s', self._hgtfile)
            hgts, _ = rio_open(self._hgtfile)
            source_is_geoid = False  # ISCE hgt files are ellipsoidal

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
            hgts = interpolateDEM(demFile, self.readLL())
            source_is_geoid = True  # GLO-30 is geoid-referenced

        if geoid_heights == source_is_geoid:
            return hgts

        # Defense-in-depth: currently unreachable, since __init__ cannot build
        # a RasterRDR without a lon_file (bounds_from_latlon_rasters raises).
        # It would become reachable if the advertised single 2-band lat/lon
        # raster is ever actually implemented.
        if self._lonfile is None:
            raise NotImplementedError(
                'Height-datum conversion requires separate lat/lon files; RasterRDR was constructed without lon_file.'
            )
        lats, lons = self.readLL()
        if geoid_heights:
            return _ellipsoidal_to_geometric(lats, lons, hgts, _WGS84_3D_CRS)
        return _geometric_to_ellipsoidal(lats, lons, hgts)


class BoundingBox(AOI):
    """Parse a bounding box AOI."""

    def __init__(self, bbox, cube_spacing_in_m: Optional[float] = None, output_directory=os.getcwd()) -> None:
        super().__init__(cube_spacing_in_m, output_directory)
        self._bounding_box = bbox
        self._type = 'bounding_box'


class GeocodedFile(AOI):
    """Parse a Geocoded file for coordinates."""

    p: RIO.Profile
    _bounding_box: BB.SNWE
    _is_dem: bool

    def __init__(
        self, path: Path, is_dem=False, cube_spacing_in_m: Optional[float] = None, output_directory=os.getcwd()
    ) -> None:
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

    def readZ(self, geoid_heights: bool = False):
        """Download a DEM for the file (or use it directly if is_dem=True).

        Heights are treated as geoid-referenced. A downloaded DEM is
        geoid-referenced at the source (GLO-30 is distributed that way and
        dem.py keeps it so), and so is the file reused when is_dem=True:
        validators.get_query_region() sets is_dem for any name starting with
        GLO/SRTM, which covers both a hand-downloaded product and the
        GLO30.dem RAiDER writes itself with that same call. Both are geoid,
        so the ambiguity that classification would otherwise create doesn't
        arise.

        By default (geoid_heights=False) returns WGS84 ellipsoidal heights,
        converted from the above. Pass geoid_heights=True to get the
        geoid-referenced heights unchanged, which is what sampling the
        weather-model cube needs.
        """
        from RAiDER.dem import download_dem
        from RAiDER.interpolator import interpolateDEM

        demFile = self._filename if self._is_dem else 'GLO30_fullres_dem.tif'
        bbox = self._bounding_box
        _, _ = download_dem(bbox, writeDEM=True, dem_path=Path(demFile))
        lats, lons = self.readLL()
        z_out = interpolateDEM(demFile, (lats, lons))

        if geoid_heights:
            return z_out
        return _geometric_to_ellipsoidal(lats, lons, z_out)


class Geocube(AOI):
    """Pull lat/lon/height from a georeferenced data cube."""

    def __init__(self, path_cube, cube_spacing_in_m: Optional[float] = None, output_directory=os.getcwd()) -> None:
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

    # untested
    def readLL(self) -> tuple[np.ndarray, np.ndarray]:
        with xr.open_dataset(self.path) as ds:
            lats = ds['latitutde'].data()
            lons = ds['longitude'].data()
        Lats, Lons = np.meshgrid(lats, lons)
        return Lats, Lons

    def readZ(self, geoid_heights: bool = False):
        """Read heights straight out of the cube.

        Unlike the other readers, this one tracks no height datum: the cube's
        'heights' variable carries no datum information and RAiDER did not
        write it. So the parameter exists only to keep the AOI.readZ()
        signature uniform -- asking for a specific datum has to fail loudly
        rather than guess, since guessing wrong displaces heights by the geoid
        undulation (~35 m in CONUS).
        """
        if geoid_heights:
            raise NotImplementedError(
                'Geocube does not record the height datum of its cube, so it '
                'cannot convert to geoid-referenced heights. Supply the '
                'heights through an AOI type that tracks a datum if the '
                'conversion is needed.'
            )

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
    snwe = (lat_stats.min, lat_stats.max, lon_stats.min, lon_stats.max)

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
