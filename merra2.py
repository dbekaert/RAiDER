"""Reader for MERRA-2 model.

Only supports pressure levels right now, that may change someday.
"""

import reader
import numpy as np
import scipy.io
import pyproj
import urllib.request


class Model(reader.Model):
    """MERRA-2 Model interface."""
    @classmethod
    def fetch(self, lats, lons, time, out):
        """Fetch MERRA-2."""
        lat_min = int(np.min(lats))
        lat_max = int(np.max(lats) + 0.5)
        lon_min = int(np.min(lons))
        lon_max = int(np.max(lons) + 0.5)
        url = _url_builder(
            time, lat_min, 1, lat_max, lon_min, 1, lon_max)
        # TODO: This doesn't work right now because the user isn't
        # authenticated with URS. Pydap provides a way to authenticate,
        # but the way the interface is set up, I'll need to write a
        # netcdf file. The most straightforward path, then, is to use
        # pydap to get the data, then write to a netcdf file ourselves.
        with urllib.request.urlopen(url) as response, open(out, 'wb') as f:
            shutil.copyfileobj(response, f)

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
