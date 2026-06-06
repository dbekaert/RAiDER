import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS

from RAiDER import utilFcns as util
from RAiDER.logger import logger
from RAiDER.models.model_levels import (
    A_137_HRES,
    B_137_HRES,
    LEVELS_25_HEIGHTS,
    LEVELS_137_HEIGHTS,
)
from RAiDER.models.weatherModel import TIME_RES, WeatherModel
from RAiDER.types import FloatArray1D, FloatArray2D, FloatArray3D


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
            out_path_combined = temp_dir / f'{out_path.stem}_combined'

            # Developed from https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5
            # All four variables are fetched in a single CDS request to halve queue wait time.
            # lnsp and z are surface-only fields; t and q span all model levels.
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
                'param': ['lnsp', 'z', 'q', 't'],
            }
            c.retrieve('reanalysis-era5-complete', params, out_path_combined)

            with xr.open_dataset(out_path_combined) as ds:
                # ERA-5 only provides z at the surface level; compute it at all
                # model levels from lnsp, t, and q via the hypsometric equation.
                # .squeeze() removes the size-1 time (and level, for lnsp/z) dims,
                # since calcgeoh expects (level, lat, lon) or (lat, lon) arrays.
                z_full, _, _ = util.calcgeoh(
                    lnsp=ds['lnsp'].values.squeeze(),
                    z_surface=ds['z'].values.squeeze(),
                    t=ds['t'].values.squeeze(),
                    q=ds['q'].values.squeeze(),
                    a=self._a,
                    b=self._b,
                    num_levels=self._levels,
                    R_d=self._R_d,
                )
                # Replace the surface-only z with the full model-level z cube,
                # broadcast to match t's (time, level, lat, lon) dimensions.
                ds_out = ds.assign(
                    z=xr.Variable(
                        dims=ds['t'].dims,
                        data=np.broadcast_to(z_full, (1, *z_full.shape)),
                    ),
                )
                ds_out.to_netcdf(out_path)

    def _batch_get_from_cds(
        self,
        times_and_paths: list[tuple[dt.datetime, Path]],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> None:
        """Download multiple ERA5 time steps in a single CDS request and split into per-datetime files."""
        import cdsapi

        c = cdsapi.Client(verify=1)

        if c.url == 'https://cds.climate.copernicus.eu/api/v2':
            logger.warning(
                'Old CDS API configuration detected: ECMWF released a breaking change in late 2024 that expired all '
                'existing credentials. This run may fail with a 404 HTTP error, in which case you may have to '
                'regenerate your CDS API credentials at https://cds.climate.copernicus.eu/how-to-api.'
            )

        # Build unique date and time lists (CDS takes the Cartesian product)
        seen_dates: dict[str, None] = {}
        seen_times: dict[str, None] = {}
        for corrected_dt, _ in times_and_paths:
            seen_dates[corrected_dt.strftime('%Y-%m-%d')] = None
            seen_times[corrected_dt.strftime('%H:%M')] = None
        date_str = '/'.join(seen_dates)
        time_str = '/'.join(seen_times)

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            batch_combined = temp_dir / 'batch_combined.nc'

            params = {
                'class': 'ea',
                'expver': '1',
                'levelist': 'all',
                'levtype': self._model_level_type,
                'stream': 'oper',
                'type': 'an',
                'date': date_str,
                'time': time_str,
                'step': '0',
                'area': [lat_max, lon_min, lat_min, lon_max],
                'grid': [0.25, 0.25],
                'format': 'netcdf',
                'param': ['lnsp', 'z', 'q', 't'],
            }
            c.retrieve('reanalysis-era5-complete', params, batch_combined)

            with xr.open_dataset(batch_combined) as ds:
                # CDS API uses 'valid_time' in newer versions, 'time' in older ones
                time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'
                for corrected_dt, out_path in times_and_paths:
                    # Strip timezone: numpy datetime64 coordinates are timezone-naive
                    target = corrected_dt.replace(tzinfo=None)
                    ds_slice = ds.sel({time_coord: target})
                    z_full, _, _ = util.calcgeoh(
                        lnsp=ds_slice['lnsp'].values.squeeze(),
                        z_surface=ds_slice['z'].values.squeeze(),
                        t=ds_slice['t'].values.squeeze(),
                        q=ds_slice['q'].values.squeeze(),
                        a=self._a,
                        b=self._b,
                        num_levels=self._levels,
                        R_d=self._R_d,
                    )
                    # Re-introduce the size-1 time dimension so the output file
                    # is identical in structure to what _get_from_cds writes.
                    ds_out = ds_slice.expand_dims(time_coord).assign(
                        z=xr.Variable(
                            dims=ds_slice['t'].expand_dims(time_coord).dims,
                            data=np.broadcast_to(z_full, (1, *z_full.shape)),
                        ),
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
        lats, lons, _, _, t, q, lnsp, z = self._makeDataCubes(Path(filename))

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z: FloatArray3D = z[::-1]
            lnsp: FloatArray2D = lnsp[::-1]
            t: FloatArray3D = t[:, ::-1]
            q: FloatArray3D = q[:, ::-1]
            lats: FloatArray1D = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z: FloatArray3D = z[..., ::-1]
            lnsp: FloatArray2D = lnsp[..., ::-1]
            t: FloatArray3D = t[..., ::-1]
            q: FloatArray3D = q[..., ::-1]
            lons: FloatArray1D = lons[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360

        geo_hgt, p, hgt = self._calculategeoh(z[0, :], lnsp, t, q)

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
    ) -> WeatherModel.DataCubes:
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
            lnsp = block['lnsp'].values[0, 0]
            z = block['z'].values.squeeze()
            lats = block['latitude'].values
            lons = block['longitude'].values

        if z.size == 0:
            raise RuntimeError('There is no data in z, you may have a problem with your mask')

        return lats, lons, lats, lons, t, q, lnsp, z
