"""Reader for MERRA-2 model.

Only supports pressure levels right now, that may change someday.
"""

import reader
import numpy as np
import scipy.io
import pyproj
import urllib.request
import json
import os.path


class Model(reader.Model):
    """MERRA-2 Model interface."""
    @classmethod
    def fetch(self, lats, lons, time, out):
        """Fetch MERRA-2."""
        # TODO: This function doesn't work right now. I'm getting a 302
        # Found response, with a message The document has moved. That's
        # annoying, it'd be nice if pydap would just follow to the new
        # page. I don't have time to debug this now, so I'll just drop
        # this comment here.
        import pydap.client
        import pydap.cas.urs

        lat_min = int(np.min(lats))
        lat_max = int(np.max(lats) + 0.5)
        lon_min = int(np.min(lons))
        lon_max = int(np.max(lons) + 0.5)
        url = ('https://goldsmr5.gesdisc.eosdis.nasa.gov:443/opendap/MERRA2/M2I6NPANA.5.12.4/1980/01/MERRA2_100.inst6_3d_ana_Np.19800101.nc4')
        config = os.path.expandvars(os.path.join('$HOME', '.urs-auth'))
        with open(config, 'r') as f:
            j = json.load(f)
            username = j['username']
            password = j['password']

        session = pydap.cas.urs.setup_session(
            username, password, check_url=url)
        # TODO: probably use a context manager so it gets closed
        dataset = pydap.client.open_url(url, session=session)

        # TODO: gotta figure out how to index as time
        timeNum = 0

        with scipy.io.netcdf.netcdf_file(out, 'w') as f:
            def index(ds):
                return ds[timeNum][:][lat_min:lat_max][lon_min:lon_max]
            t = f.createVariable('T', float, [])
            t[:] = index(dataset['T'])
            lats = f.createVariable('lat', float, [])
            lats[:] = dataset['lat'][lat_min:lat_max]
            lons = f.createVariable('lon', float, [])
            lons[:] = dataset['lon'][lon_min:lon_max]
            q = f.createVariable('QV', float, [])
            q[:] = index(dataset['QV'])
            z = f.createVariable('H', float, [])
            z[:] = index(dataset['H'])
            p = f.createVariable('lev', float, [])
            p[:] = dataset['lev'][:]

    @classmethod
    def load_pressure_level(self, filename):
        with scipy.io.netcdf.netcdf_file(
                filename, 'r', maskandscale=True) as f:
            lats = f.variables['lat'][:].copy()
            lons = f.variables['lon'][:].copy()
            t = f.variables['T'][0].copy()
            q = f.variables['QV'][0].copy()
            z = f.variables['H'][0].copy()
            p = f.variables['lev'][0].copy()
        proj = pyproj.Proj('lla')
        return xs, ys, proj, t, q, z, p

    @classmethod
    def weather_and_nodes(self, filename):
        return self.weather_and_nodes_from_pressure_levels(filename)


def _url_builder(time, lat_min, lat_step, lat_max, lon_min, lon_step, lon_max):
    if lon_max < 0:
        lon_max += 360
    if lon_min < 0:
        lon_min += 360
    if lon_min > lon_max:
        lon_max, lon_min = lon_min, lon_max
    timeStr = '[0]'  # TODO: change
    latStr = f'[{lat_min}:{lat_step}:{lat_max}]'
    lonStr = f'[{lon_min}:{lon_step}:{lon_max}]'  # TODO: wrap
    lvs = '[0:1:41]'
    combined = f'{timeStr}{lvs}{latStr}{lonStr}'
    return ('https://goldsmr5.gesdisc.eosdis.nasa.gov:443/opendap/MERRA2/'
            f'M2I6NPANA.5.12.4/{time.strftime("%Y/%m")}/'
            f'MERRA2_100.inst6_3d_ana_Np.{time.strftime("%Y%m%d")}.nc4.nc?'
            f'T{combined},H{combined},lat{latStr},lev{lvs},lon{lonStr},'
            f'time{timeStr}')
