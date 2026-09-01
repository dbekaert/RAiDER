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
        # round to the closest legal time
        corrected_DT = util.round_date(acqTime, dt.timedelta(hours=self._time_res))
        if not corrected_DT == acqTime:
            logger.warning('Rounded given datetime from  %s to %s', acqTime, corrected_DT)

        # A single date is a one-element batch, so both level types go through the
        # batch writer. That keeps one download path per level type rather than two
        # that can drift apart: in particular the model-level branch requests the
        # surface fields (lnsp and z, archived at model level 1 only) separately from
        # t/q, which CDS will not return alongside each other in a single netCDF.
        self._batch_get_from_cds([(corrected_DT, out_path)], lat_min, lat_max, lon_min, lon_max)

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

        if self._model_level_type == 'ml':
            dataset = 'reanalysis-era5-complete'
            # Only the model-level branch uses these; the pressure-level branch builds
            # its own request per datetime below.
            base_params = {
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
            }
        else:
            dataset = 'reanalysis-era5-pressure-levels'
            param = ['z', 't', 'q']

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            surface_file = temp_dir / 'batch_surface.nc'
            ml_file = temp_dir / 'batch_ml.nc'
            pl_file = temp_dir / 'batch_pl.nc'

            if self._model_level_type == 'pl':
                # MARS for reanalysis-era5-pressure-levels allows only one date per month
                # per request; multiple same-month dates cause "Duplicate value for month".
                # The CDS API v3 also auto-splits a list by month before sending to MARS,
                # so batching is not possible. Issue one request per datetime.
                for corrected_dt, out_path in times_and_paths:
                    pl_params = {
                        'product_type': 'reanalysis',
                        'levelist': 'all',
                        'levtype': 'pl',
                        'date': corrected_dt.strftime('%Y-%m-%d'),
                        'time': corrected_dt.strftime('%H:%M'),
                        'area': [lat_max, lon_min, lat_min, lon_max],
                        'data_format': 'netcdf',
                        'param': param,
                    }
                    c.retrieve(dataset, pl_params, pl_file)

                    with xr.open_dataset(pl_file) as ds:
                        tc = 'valid_time' if 'valid_time' in ds.coords else 'time'
                        target = corrected_dt.replace(tzinfo=None)
                        # Resolve the level dimension by name and normalise the axis
                        # order the same way _load_pressure_level does, rather than
                        # assuming a positional layout: a positional transpose swaps
                        # lat/lon identically across z/t/q, so a differently ordered
                        # response produces garbage with no shape error to catch it.
                        lev_dim = 'pressure_level' if 'pressure_level' in ds.dims else 'level'
                        ds_slice = ds.isel({tc: 0}).transpose('latitude', 'longitude', lev_dim, ...)

                        z_v = (ds_slice['z'].values.squeeze() / self._g0)[np.newaxis]
                        t_v = ds_slice['t'].values.squeeze()[np.newaxis]
                        q_v = ds_slice['q'].values.squeeze()[np.newaxis]

                        ds_out = xr.Dataset(
                            {
                                # z is written as geopotential HEIGHT (already divided
                                # by g0). The units attribute records that so
                                # _load_pressure_level does not have to infer it.
                                'z': xr.Variable(
                                    (tc, 'latitude', 'longitude', 'pressure_level'),
                                    z_v,
                                    {'units': 'm'},
                                ),
                                't': xr.Variable((tc, 'latitude', 'longitude', 'pressure_level'), t_v),
                                'q': xr.Variable((tc, 'latitude', 'longitude', 'pressure_level'), q_v),
                            },
                            coords={
                                tc: np.array([target], dtype='datetime64[ns]'),
                                'pressure_level': ds_slice[lev_dim].values,
                                'latitude': ds_slice['latitude'].values,
                                'longitude': ds_slice['longitude'].values,
                            },
                        )
                        ds_out.to_netcdf(out_path)
            else:
                # lnsp and z are surface fields, archived at model level 1 only; t and q
                # span all model levels. CDS returns a mixed-level request as separate
                # files rather than as one netCDF, so the two groups have to be requested
                # independently. That is two requests instead of one, but still far better
                # than a separate pair of requests for every datetime.
                c.retrieve(dataset, {**base_params, 'param': ['lnsp', 'z']}, surface_file)
                c.retrieve(dataset, {**base_params, 'param': ['t', 'q']}, ml_file)

                with xr.open_dataset(surface_file) as ds_surface, xr.open_dataset(ml_file) as ds_ml:
                    # CDS API uses 'valid_time' in newer versions, 'time' in older ones
                    tc_s = 'valid_time' if 'valid_time' in ds_surface.coords else 'time'
                    tc_m = 'valid_time' if 'valid_time' in ds_ml.coords else 'time'
                    # Same normalisation as the pressure-level branch: resolve the level
                    # dimension by name and put the axes in the order the writer below
                    # declares, instead of trusting the response's on-disk layout.
                    lev_dim = 'model_level' if 'model_level' in ds_ml.dims else 'level'
                    ds_ml = ds_ml.transpose(..., lev_dim, 'latitude', 'longitude')

                    for corrected_dt, out_path in times_and_paths:
                        # Strip timezone: numpy datetime64 coordinates are timezone-naive
                        target = corrected_dt.replace(tzinfo=None)
                        surf_slice = ds_surface.sel({tc_s: target})
                        ml_slice = ds_ml.sel({tc_m: target})

                        lnsp_v = surf_slice['lnsp'].values.squeeze()   # (nlat, nlon)
                        z_sfc_v = surf_slice['z'].values.squeeze()     # (nlat, nlon)
                        t_v = ml_slice['t'].values.squeeze()           # (nlev, nlat, nlon)
                        q_v = ml_slice['q'].values.squeeze()           # (nlev, nlat, nlon)

                        z_full, _, _ = util.calcgeoh(
                            lnsp=lnsp_v,
                            z_surface=z_sfc_v,
                            t=t_v,
                            q=q_v,
                            a=self._a,
                            b=self._b,
                            num_levels=self._levels,
                            R_d=self._R_d,
                        )

                        # Write a single-time file matching the structure _makeDataCubes
                        # expects: t/q/z as (time, model_level, lat, lon) and lnsp as
                        # (time, sfc_level, lat, lon) so that [0,0] indexing gives (lat,lon).
                        ds_out = xr.Dataset(
                            {
                                't':    xr.Variable((tc_m, 'model_level', 'latitude', 'longitude'), t_v[np.newaxis]),
                                'q':    xr.Variable((tc_m, 'model_level', 'latitude', 'longitude'), q_v[np.newaxis]),
                                'z':    xr.Variable((tc_m, 'model_level', 'latitude', 'longitude'), z_full[np.newaxis]),
                                'lnsp': xr.Variable((tc_m, 'sfc_level', 'latitude', 'longitude'), lnsp_v[np.newaxis, np.newaxis]),
                            },
                            coords={
                                tc_m: np.array([target], dtype='datetime64[ns]'),
                                'model_level': ml_slice[lev_dim].values,
                                'latitude': ml_slice['latitude'].values,
                                'longitude': ml_slice['longitude'].values,
                            },
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

        # data ordering
        if lats[0] > lats[1]:
            # z is (level, lat, lon). It needs BOTH axes reversed: the latitude
            # axis to match t/q/lnsp, and the level axis because the surface
            # geopotential is taken as z[0] below (the stored cube runs
            # top-of-atmosphere -> surface). Reversing only axis 0, as this
            # previously did, left the height field mirrored north-south
            # relative to the meteorology.
            z: FloatArray3D = z[::-1, ::-1]
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
        with xr.open_dataset(filename) as ds:
            # Drop the singleton time dimension (name varies by CDS API version)
            for time_dim in ('valid_time', 'time'):
                if time_dim in ds.dims:
                    ds = ds.isel({time_dim: 0})
                    break

            lev_dim = 'pressure_level' if 'pressure_level' in ds.dims else 'level'
            # Normalize dimension order by name so the file's on-disk layout
            # doesn't matter
            # The ellipsis absorbs any dim not listed here (ERA-5T responses sometimes
            # carry an extra 'expver'); without it transpose demands a permutation of
            # every dim and raises ValueError.
            ds = ds.transpose('latitude', 'longitude', lev_dim, ...)

            z = ds['z'].values.astype(np.float64)
            t = ds['t'].values
            q = ds['q'].values
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            levels = ds[lev_dim].values * 100  # hPa -> Pa
            z_units = ds['z'].attrs.get('units', '')

        # Files written by _batch_get_from_cds store z as geopotential height in
        # metres and record that in the units attribute; raw CDS files store
        # geopotential in m**2 s**-2. Trust the units when present -- the
        # magnitude fallback below misclassifies any file whose levels all sit
        # below ~10 km.
        if z_units:
            is_geopotential = 'm**2' in z_units or 'm2' in z_units
        else:
            # 100,000 m**2 s**-2 is only ~10.2 km of geopotential height, so this
            # is wrong for a request restricted to near-surface pressure levels.
            is_geopotential = np.nanmax(z) > 100_000
            logger.warning(
                'Pressure-level file %s records no units for z; inferring %s from its magnitude.',
                filename,
                'geopotential' if is_geopotential else 'geopotential height',
            )

        if is_geopotential:
            z = z / self._g0

        # Reorder axes (consistently across all cubes) so lats and lons are
        # ascending and levels go surface -> TOA
        if lats[0] > lats[1]:
            z = z[::-1]
            t = t[::-1]
            q = q[::-1]
            lats = lats[::-1]
        if lons[0] > lons[1]:
            z = z[:, ::-1]
            t = t[:, ::-1]
            q = q[:, ::-1]
            lons = lons[::-1]
        if levels[0] < levels[-1]:
            z = z[..., ::-1]
            t = t[..., ::-1]
            q = q[..., ::-1]
            levels = levels[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360

        self._t = t
        self._q = q

        # re-assign lons, lats to match heights
        self._lons, self._lats = np.meshgrid(lons, lats)

        # correct heights for latitude
        self._get_heights(self._lats, z)

        self._p = np.broadcast_to(levels[np.newaxis, np.newaxis, :], self._zs.shape)

        self._ys = self._lats.copy()
        self._xs = self._lons.copy()

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
