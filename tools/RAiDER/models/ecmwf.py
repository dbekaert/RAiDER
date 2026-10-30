import datetime as dt
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from pyproj import CRS

from RAiDER import utilFcns as util
from RAiDER.constants import _g0
from RAiDER.logger import logger
from RAiDER.models.model_levels import (
    A_137_HRES,
    B_137_HRES,
    LEVELS_25_HEIGHTS,
    LEVELS_137_HEIGHTS,
)
from RAiDER.models.weatherModel import TIME_RES, WeatherModel
from RAiDER.types import FloatArray1D, FloatArray2D, FloatArray3D


# How far (in m) a file's reconstructed lowest model level may sit from the stored
# one before it is flagged as storing virtual temperature as t (#819). Correct
# files match exactly; stale ones are off by centimetres on real scenes.
_STALE_T_TOLERANCE_M = 1e-3


def _as_surface_field(field: xr.DataArray) -> Optional[FloatArray2D]:
    """Return the (lat, lon) field if ``field`` holds a single surface field, else None.

    ECMWF model-level files store surface fields (lnsp, surface z) in several
    layouts: with no level axis, with a size-1 level axis of their own, or on
    the shared level axis with the field on one level and NaN on the rest (or
    the same field repeated on every level). All of these are recognized. A
    full model-level cube, whose levels differ, is not a surface field.

    Only the latitude and longitude axes are kept by name; every other axis
    (time, level, whatever a converter adds) is folded into one leading axis
    and inspected. Squeezing instead would also drop a latitude or longitude
    axis of length one, which a small query domain can produce.

    Every slice along that leading axis must be either all-NaN padding or a
    fully finite field, and the finite ones must all be the same field. A
    slice that is only partly finite, or that holds Inf, is neither, so the
    variable is rejected rather than read from whichever slices happen to be
    complete: a file whose levels disagree must not load from one of them.
    """
    leading = [dim for dim in field.dims if dim not in ('latitude', 'longitude')]
    field = field.transpose(*leading, 'latitude', 'longitude')
    values = field.values.reshape(-1, *field.shape[-2:])
    flat = values.reshape(values.shape[0], -1)
    complete = np.isfinite(flat).all(axis=1)
    padding = np.isnan(flat).all(axis=1)
    if not complete.any() or not (complete | padding).all():
        return None
    candidates = values[complete]
    if np.array_equal(candidates, np.broadcast_to(candidates[0], candidates.shape)):
        return candidates[0].copy()
    return None


class ECMWF(WeatherModel):
    """Implement ECMWF models."""

    def __init__(self) -> None:
        super().__init__()

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        self._time_res = TIME_RES['ECMWF']

        self._lon_res = 0.25
        self._lat_res = 0.25
        self._proj = CRS.from_epsg(4326)

        self._model_level_type = 'ml'  # Default

    def __pressure_levels__(self):
        self._zlevels = np.flipud(LEVELS_25_HEIGHTS)
        self._levels = len(self._zlevels)

    def __model_levels__(self):
        self._levels = 137
        self._zlevels = np.flipud(LEVELS_137_HEIGHTS)
        self._a = A_137_HRES
        self._b = B_137_HRES

    def load_weather(self, filename=None, *args, **kwargs) -> None:
        """
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        """
        filename = filename if filename is not None else self.files[0]
        self._load_model_level(filename)

    def _fetch(self, out: Path) -> None:
        """Fetch a weather model from ECMWF."""
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._ll_bounds
        # execute the search at ECMWF
        self._get_from_ecmwf(lat_min, lat_max, self._lat_res, lon_min, lon_max, self._lon_res, self._time, out)

    def _get_from_ecmwf(self, lat_min, lat_max, lat_step, lon_min, lon_max, lon_step, time, out: Path) -> None:
        import ecmwfapi

        server = ecmwfapi.ECMWFDataServer()

        corrected_DT = util.round_date(time, dt.timedelta(hours=self._time_res))
        if not corrected_DT == time:
            logger.warning('Rounded given datetime from  %s to %s', time, corrected_DT)

        server.retrieve(
            {
                'class': self._classname,  # HRES
                'dataset': self._dataset,
                'expver': f'{self._expver}',
                # They warn me against all, but it works well
                'levelist': 'all',
                'levtype': 'ml',  # Model levels
                'param': 'lnsp/q/z/t',  # Necessary variables
                'stream': 'oper',
                # date: Specify a single date as "2015-08-01" or a period as
                # "2015-08-01/to/2015-08-31".
                'date': corrected_DT.strftime('%Y-%m-%d'),
                # type: Use an (analysis) unless you have a particular reason to
                # use fc (forecast).
                'type': 'an',
                # time: With type=an, time can be any of
                # "00:00:00/06:00:00/12:00:00/18:00:00".  With type=fc, time can
                # be any of "00:00:00/12:00:00",
                'time': corrected_DT.strftime('%H:%M:%S'),
                # step: With type=an, step is always "0". With type=fc, step can
                # be any of "3/6/9/12".
                'step': '0',
                # grid: Only regular lat/lon grids are supported.
                'grid': f'{lat_step}/{lon_step}',
                'area': f'{lat_max}/{lon_min}/{lat_min}/{lon_max}',  # area: N/W/S/E
                'format': 'netcdf',
                'resol': 'av',
                'target': str(out),  # target: the name of the output file.
            }
        )

    def _get_from_cds(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        acqTime: dt.datetime,
        out_path: Path,
    ) -> None:
        """Used for ERA5."""
        import cdsapi

        c = cdsapi.Client(verify=1)

        if c.url == 'https://cds.climate.copernicus.eu/api/v2':
            logger.warning(
                'Old CDS API configuration detected: ECMWF released a breaking change in late 2024 that expired all '
                'existing credentials. This run may fail with a 404 HTTP error, in which case you may have to '
                'regenerate your CDS API credentials at https://cds.climate.copernicus.eu/how-to-api.'
            )

        # round to the closest legal time
        corrected_DT = util.round_date(acqTime, dt.timedelta(hours=self._time_res))
        if not corrected_DT == acqTime:
            logger.warning('Rounded given datetime from  %s to %s', acqTime, corrected_DT)

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            out_path_lnsp_z = temp_dir / f'{out_path.stem}_lnsp_z'
            out_path_t_q = temp_dir / f'{out_path.stem}_t_q'

            # Developed from https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5
            params = {
                'class': 'ea',
                'expver': '1',
                'levelist': 'all',
                'levtype': self._model_level_type,  # 'ml' for model levels or 'pl' for pressure levels
                'stream': 'oper',
                'type': 'an',
                'date': corrected_DT.strftime('%Y-%m-%d'),
                'time': corrected_DT.strftime('%H:%M'),
                # step: With type=an, step is always "0". With type=fc, step can
                # be any of "3/6/9/12".
                'step': '0',
                'area': [lat_max, lon_min, lat_min, lon_max],
                'grid': [0.25, 0.25],
                'format': 'netcdf',
            }
            # lnsp and z are surface-only fields; t and q span all model levels. These
            # must be two separate requests: CDS returns a mixed-level request as
            # separate files rather than as one netCDF, and even where it does return a
            # single file, the surface fields come back padded onto the full level axis,
            # which breaks the (lat, lon) shape calcgeoh and _makeDataCubes expect.
            params['param'] = ['lnsp', 'z']
            c.retrieve('reanalysis-era5-complete', params, out_path_lnsp_z)
            params['param'] = ['q', 't']
            c.retrieve('reanalysis-era5-complete', params, out_path_t_q)

            with xr.open_dataset(out_path_lnsp_z) as ds_lnsp_z, xr.open_dataset(out_path_t_q) as ds_t_q:
                # ERA-5 only provides z at the surface level; compute it at all
                # model levels from lnsp, t, and q via the hypsometric equation.
                # .squeeze() removes the size-1 time dim (and, for lnsp/z, the
                # size-1 level dim), since calcgeoh expects (level, lat, lon) or
                # (lat, lon) arrays.
                z_full, _, _ = util.calcgeoh(
                    lnsp=ds_lnsp_z['lnsp'].values.squeeze(),
                    z_surface=ds_lnsp_z['z'].values.squeeze(),
                    t=ds_t_q['t'].values.squeeze(),
                    q=ds_t_q['q'].values.squeeze(),
                    a=self._a,
                    b=self._b,
                    num_levels=self._levels,
                    R_d=self._R_d,
                )
                # Replace the surface-only z with the full model-level z cube,
                # broadcast to match t's (time, level, lat, lon) dimensions.
                ds_out = ds_t_q.assign(
                    z=xr.Variable(
                        dims=ds_t_q['t'].dims,
                        data=np.broadcast_to(z_full, (1, *z_full.shape)),
                    ),
                    # xarray aligns lnsp and z_surface on the level coordinate,
                    # so they land on the first level and are NaN on every other
                    # one. _makeDataCubes recognizes that layout (among others)
                    # and recovers the (lat, lon) fields.
                    lnsp=ds_lnsp_z['lnsp'],
                    # Also keep the downloaded surface geopotential: the loader
                    # needs it to start the hydrostatic integration at the
                    # terrain surface rather than at the lowest full level,
                    # which sits roughly 10 m above it. It inherits the CDS
                    # attrs of z, so give it a long_name of its own.
                    z_surface=ds_lnsp_z['z'].assign_attrs(long_name='Surface geopotential'),
                )
                ds_out.to_netcdf(out_path)

    def _download_ecmwf(self, lat_min, lat_max, lat_step, lon_min, lon_max, lon_step, time, out: Path) -> None:
        """Used for HRES."""
        from ecmwfapi import ECMWFService

        server = ECMWFService('mars')

        # round to the closest legal time
        corrected_DT = util.round_date(time, dt.timedelta(hours=self._time_res))
        if not corrected_DT == time:
            logger.warning('Rounded given datetime from  %s to %s', time, corrected_DT)

        if self._model_level_type == 'ml':
            param = '129/130/133/152'
        else:
            param = '129.128/130.128/133.128/152'

        server.execute(
            {
                'class': self._classname,
                'dataset': self._dataset,
                'expver': f'{self._expver}',
                'resol': 'av',
                'stream': 'oper',
                'type': 'an',
                'levelist': 'all',
                'levtype': f'{self._model_level_type}',
                'param': param,
                'date': dt.datetime.strftime(corrected_DT, '%Y-%m-%d'),
                'time': dt.time.strftime(corrected_DT.time(), '%H:%M'),
                'step': '0',
                'grid': f'{lon_step}/{lat_step}',
                'area': f'{lat_max}/{util.floorish(lon_min, 0.1)}/{util.floorish(lat_min, 0.1)}/{lon_max}',
                'format': 'netcdf',
            },
            str(out),
        )

    def _load_model_level(self, filename, *args, **kwargs) -> None:
        # read data from netcdf file
        lats, lons, _, _, t, q, lnsp, z, z_surface = self._makeDataCubes(Path(filename))

        # The stored lowest full level, to check the loaded t against below. Only
        # files whose z is a genuine full-level cube alongside a separate surface
        # field can be checked: the fallback starts the integration at that level,
        # so it would match trivially, and surface-only layouts (z padded with NaN,
        # or the surface field repeated on every level) have no full level stored.
        is_full_cube = z.shape == t.shape and np.all(np.isfinite(z[-1]))
        has_separate_surface = z_surface is not None and is_full_cube and not np.array_equal(z[-1], z_surface)
        z_lowest_stored = z[-1] if has_separate_surface else None

        if z_surface is None:
            # Files fetched by v0.6.0 up to this change carry only the
            # recomputed full-level cube, stored top-of-atmosphere -> surface:
            # fall back to its lowest full level, which sits roughly 10 m
            # above the terrain, so every column comes out shifted up by that
            # amount. Taken here, before the orientation fixes below, so the
            # fallback does not depend on which axes get reversed. Every file
            # in this layout was also written by a fetch that stored virtual
            # temperature as t (#819).
            logger.warning(
                '%s has no surface geopotential (fetched before it was stored). Starting the '
                'hydrostatic integration at the lowest full model level, ~10 m above the terrain, '
                'which raises every column by that amount and biases delays by roughly +3 mm. '
                'It also stores virtual temperature as t (#819), so temperature, humidity and '
                'delays computed from it are biased too. Delete the file to re-download it.',
                filename,
            )
            z_surface = z[-1]

        # ECMWF appears to give me this backwards. The surface geopotential
        # has to be reversed on the same axes as t/q/lnsp, or the heights end
        # up mirrored relative to the meteorology, and so does the stored
        # lowest level that the reconstruction is checked against.
        if lats[0] > lats[1]:
            lnsp: FloatArray2D = lnsp[::-1]
            t: FloatArray3D = t[:, ::-1]
            q: FloatArray3D = q[:, ::-1]
            lats: FloatArray1D = lats[::-1]
            z_surface = z_surface[::-1]
            if z_lowest_stored is not None:
                z_lowest_stored = z_lowest_stored[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            lnsp: FloatArray2D = lnsp[..., ::-1]
            t: FloatArray3D = t[..., ::-1]
            q: FloatArray3D = q[..., ::-1]
            lons: FloatArray1D = lons[::-1]
            z_surface = z_surface[..., ::-1]
            if z_lowest_stored is not None:
                z_lowest_stored = z_lowest_stored[..., ::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360

        geo_hgt, p, hgt = self._calculategeoh(z_surface, lnsp, t, q)
        if not np.all(np.isfinite(hgt)):
            raise RuntimeError(f'{filename}: model-level heights contain non-finite values; unrecognized file layout?')

        if z_lowest_stored is not None:
            # The fetch computed the stored z cube with this same integration from
            # the same inputs, so a file whose t is real temperature reproduces its
            # lowest level exactly. Files fetched between #820 and the fix for #819
            # store virtual temperature as t; integrating that applies the moisture
            # factor twice and lands above the stored level, by centimetres on real
            # scenes.
            offset = float(np.nanmax(np.abs(geo_hgt[-1] - z_lowest_stored))) / _g0
            if offset > _STALE_T_TOLERANCE_M:
                logger.warning(
                    '%s stores virtual temperature as t (fetched before the fix for #819): its '
                    'reconstructed lowest model level is %.3f m off the stored one, and temperature, '
                    'humidity and delays computed from it are biased. Delete the file to re-download it.',
                    filename,
                    offset,
                )

        self._lons, self._lats = np.meshgrid(lons, lats)

        # ys is latitude
        self._get_heights(self._lats, hgt.transpose(1, 2, 0))  # sets self._zs

        # We want to support both pressure levels and true pressure grids.
        # If the shape has one dimension, we'll scale it up to act as a
        # grid, otherwise we'll leave it alone.
        if p.ndim == 1:
            p = np.broadcast_to(p[:, np.newaxis, np.newaxis], self._zs.shape)

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        p = p.transpose(1, 2, 0)
        t = t.transpose(1, 2, 0)
        q = q.transpose(1, 2, 0)

        # Flip all the axes so that zs are in order from bottom to top
        # lats / lons are simply replicated to all heights so they don't need flipped
        self._p = np.flip(p, axis=2)
        self._t = np.flip(t, axis=2)
        self._q = np.flip(q, axis=2)
        self._ys = self._lats.copy()
        self._xs = self._lons.copy()
        self._zs = np.flip(self._zs, axis=2)

    def _load_pressure_level(self, filename) -> None:
        with xr.open_dataset(filename) as block:
            # Pull the data
            z = np.squeeze(block['z'].values)
            t = np.squeeze(block['t'].values)
            q = np.squeeze(block['q'].values)
            lats = np.squeeze(block['latitude'].values)
            lons = np.squeeze(block['longitude'].values)
            levels = np.squeeze(block['level'].values) * 100

        z = np.flip(z, axis=1)

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z = z[::-1]
            t = t[:, ::-1]
            q = q[:, ::-1]
            lats = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z = z[..., ::-1]
            t = t[..., ::-1]
            q = q[..., ::-1]
            lons = lons[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360

        self._t = t
        self._q = q

        geo_hgt = (z / self._g0).transpose(1, 2, 0)

        # re-assign lons, lats to match heights
        self._lons, self._lats = np.meshgrid(lons, lats)

        # correct heights for latitude
        self._get_heights(self._lats, geo_hgt)

        self._p = np.broadcast_to(levels[np.newaxis, np.newaxis, :], self._zs.shape)

        # Re-structure from (heights, lats, lons) to (lons, lats, heights)
        self._t = self._t.transpose(1, 2, 0)
        self._q = self._q.transpose(1, 2, 0)
        self._ys = self._lats.copy()
        self._xs = self._lons.copy()

        # flip z to go from surface to toa
        self._p = np.flip(self._p, axis=2)
        self._t = np.flip(self._t, axis=2)
        self._q = np.flip(self._q, axis=2)

    def _makeDataCubes(
        self,
        path: Path,
        verbose: bool = False,
    ) -> tuple[
        FloatArray1D,  # lats
        FloatArray1D,  # lons
        FloatArray1D,  # xs
        FloatArray1D,  # ys
        FloatArray3D,  # t
        FloatArray3D,  # q
        FloatArray2D,  # lnsp
        FloatArray3D,  # z - full model-level geopotential cube
        Optional[FloatArray2D],  # z_surface - surface geopotential, None for files that only carry the full-level cube
    ]:
        """
        Create a cube of data representing temperature and relative humidity
        at specified pressure levels.
        """
        # get ll_bounds
        S, N, W, E = self._ll_bounds

        with xr.open_dataset(path) as ds:
            ds = ds.assign_coords(longitude=(((ds['longitude'] + 180) % 360) - 180))

            # mask based on query bounds
            m1 = (S <= ds['latitude']) & (N >= ds['latitude'])
            m2 = (W <= ds['longitude']) & (E >= ds['longitude'])
            block = ds.where(m1 & m2, drop=True)

            # Pull the data
            t = block['t'].values.squeeze()
            q = block['q'].values.squeeze()
            z = block['z'].values.squeeze()
            # The surface fields are keyed on their stored layout, not on the
            # model that wrote the file: ERA-5 and HRES files from different
            # RAiDER versions and converters all go through this reader.
            lnsp = _as_surface_field(block['lnsp'])
            if lnsp is None:
                raise ValueError(f'{path}: cannot recover a (lat, lon) surface field from lnsp {block["lnsp"].dims}')
            if 'z_surface' in block:
                # Files fetched since the surface geopotential was preserved
                # alongside the recomputed full-level z cube.
                z_surface = _as_surface_field(block['z_surface'])
                if z_surface is None:
                    raise ValueError(
                        f'{path}: cannot recover a (lat, lon) surface field from z_surface {block["z_surface"].dims}'
                    )
            else:
                # Files fetched before v0.6.0 (and any other file that stores
                # the downloaded geopotential as-is) carry the surface field
                # itself in z. A recomputed full-level cube is not a surface
                # field, so this is None for files fetched by v0.6.0 up to
                # this change, and the loader falls back.
                z_surface = _as_surface_field(block['z'])
                if z_surface is None and (z.shape != t.shape or not np.all(np.isfinite(z[-1]))):
                    # The loader will fall back to the lowest full level of
                    # z, so z has to be a complete model-level cube for that
                    # to mean anything. A cube that is neither a surface
                    # field nor whole is an unrecognized layout.
                    raise ValueError(
                        f'{path}: z is neither a surface field nor a full model-level cube '
                        f'(shape {z.shape}, t shape {t.shape}); cannot start the integration from it'
                    )
            if z_surface is not None and z_surface.shape != lnsp.shape:
                raise ValueError(f'{path}: surface geopotential shape {z_surface.shape} != lnsp shape {lnsp.shape}')
            lats = block['latitude'].values
            lons = block['longitude'].values

        if z.size == 0:
            raise RuntimeError('There is no data in z, you may have a problem with your mask')

        return lats, lons, lats, lons, t, q, lnsp, z, z_surface
