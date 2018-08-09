"""
Read model level WRF ECMWF file.
"""


import datetime
import numpy as np
import pyproj
import scipy.io
import reader

try:
    import ecmwfapi
except ImportError as e:
    ecmwfapi_error = e


class Model(reader.Model):
    time_period = datetime.timedelta(hours=6)

    @classmethod
    def load_model_level(self, fname):
        with scipy.io.netcdf.netcdf_file(fname, 'r', maskandscale=True) as f:
            # 0,0 to get first time and first level
            z = f.variables['z'][0][0].copy()
            lnsp = f.variables['lnsp'][0][0].copy()
            t = f.variables['t'][0].copy()
            q = f.variables['q'][0].copy()
            lats = f.variables['latitude'][:].copy()
            lons = f.variables['longitude'][:].copy()

        lla = pyproj.Proj(proj='latlong')

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z = z[::-1]
            lnsp = lnsp[::-1]
            t = t[:, ::-1]
            q = q[:, ::-1]
            lats = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z = z[..., ::-1]
            lnsp = lnsp[..., ::-1]
            t = t[..., ::-1]
            q = q[..., ::-1]
            lons = lons[::-1]

        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360

        return lons, lats, lla, t, q, z, lnsp

    @classmethod
    def get_from_ecmwf(self, lat_min, lat_max, lat_step, lon_min, lon_max,
                       lon_step, time, out):
        try:
            ecmwfapi
        except NameError:
            raise ecmwfapi_error

        server = ecmwfapi.ECMWFDataServer()

        corrected_date = round_date(time, datetime.timedelta(hours=6))

        server.retrieve({
            "class": self.classname,  # ERA-Interim
            'dataset': self.dataset,
            "expver": "1",
            # They warn me against all, but it works well
            "levelist": 'all',
            "levtype": "ml",  # Model levels
            "param": "lnsp/q/z/t",  # Necessary variables
            "stream": "oper",
            # date: Specify a single date as "2015-08-01" or a period as
            # "2015-08-01/to/2015-08-31".
            "date": datetime.datetime.strftime(corrected_date, "%Y-%m-%d"),
            # type: Use an (analysis) unless you have a particular reason to
            # use fc (forecast).
            "type": "an",
            # time: With type=an, time can be any of
            # "00:00:00/06:00:00/12:00:00/18:00:00".  With type=fc, time can
            # be any of "00:00:00/12:00:00",
            "time": datetime.datetime.strftime(corrected_date, "%H:%M:%S"),
            # step: With type=an, step is always "0". With type=fc, step can
            # be any of "3/6/9/12".
            "step": "0",
            # grid: Only regular lat/lon grids are supported.
            "grid": f'{lat_step}/{lon_step}',
            "area": f'{lat_max}/{lon_min}/{lat_min}/{lon_max}',    # area: N/W/S/E
            "format": "netcdf",
            "resol": "av",
            "target": out,    # target: the name of the output file.
        })

    @classmethod
    def fetch(self, lats, lons, time, out):
        lat_min = np.min(lats)
        lat_max = np.max(lats)
        lon_min = np.min(lons)
        lon_max = np.max(lons)
        lat_res = 0.2
        lon_res = 0.2

        self.get_from_ecmwf(
                lat_min, lat_max, lat_res, lon_min, lon_max, lon_res, time,
                out)


def round_date(date, precision):
    # First try rounding up
    # Timedelta since the beginning of time
    datedelta = datetime.datetime.min - date
    # Round that timedelta to the specified precision
    rem = datedelta % precision
    # Add back to get date rounded up
    round_up = date + rem

    # Next try rounding down
    datedelta = date - datetime.datetime.min
    rem = datedelta % precision
    round_down = date - rem

    # It's not the most efficient to calculate both and then choose, but
    # it's clear, and performance isn't critical here.
    up_diff = round_up - date
    down_diff = date - round_down

    return round_up if up_diff < down_diff else round_down
