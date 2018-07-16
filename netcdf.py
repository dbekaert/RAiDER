"""Netcdf reader for weather data."""


# TODO: use a different method of interpolation (someone told me linear
# is inaccurate).
# TODO: (maybe) add another layer below to make sure we get everything
# What did he mean by this? ^^


import numpy as np
import pyproj
import reader
import scipy.interpolate as interpolate
import scipy.io.netcdf as netcdf
import util


# NetCDF files have the ability to record their nodata value, but in the
# particular NetCDF files that I'm reading, this field is left
# unspecified and a nodata value of -999 is used. The solution I'm using
# is to check if nodata is specified, and otherwise assume it's -999.
_default_fill_value = -999


# Parameters from Hanssen, 2001
_k1 = 0.776 # [K/Pa]
# Should be k2'
_k2 = 0.233 # [K/Pa]
_k3 = 3.75e3 # [K^2/Pa]


def _read_netcdf(out, plev, scipy_interpolate):
    """Return a NetCDFModel given open netcdf files."""
    # n.b.: all of these things we read are arrays of length 1, so
    # we get the first element to access the actual data.
    lats = out.variables['XLAT'][0]
    # Why is it XLONG with a G? Maybe the G means geo (but then why
    # isn't it XLATG?).
    lons = out.variables['XLONG'][0]

    # TODO: it'd be cool to use some kind of units package
    if plev.variables['P_PL'].units.decode('utf-8') == 'Pa':
        plevs = plev.variables['P_PL'][0]
    else:
        raise NetCDFException(
                f"Unknown units for pressure: "
                "'{plev.variables['P_PL'].units}'")

    if plev.variables['T_PL'].units.decode('utf-8') == 'K':
        temps = plev.variables['T_PL']
    else:
        err = "Unknown units for temperature: '{}'"
        raise NetCDFException(err.format(plev.variables['T_PL'].units))

    # TODO: extract partial pressure directly (q?)
    if plev.variables['RH_PL'].units.decode('utf-8') == '%':
        humids = plev.variables['RH_PL']
    else:
        err = "Unknown units for relative humidity: '{}'"
        raise NetCDFException(err.format(plev.variables['RH_PL'].units))

    if plev.variables['GHT_PL'].units.decode('utf-8') == 'm':
        geopotential_heights = plev.variables['GHT_PL']
    else:
        err = "Unknown units for geopotential height: '{}'"
        raise NetCDFException(err.format(plev.variables['GHT_PL'].units))

    # _FillValue is not always set, but when it is we want to read it
    try:
        temp_fill = temps._FillValue
    except AttributeError:
        temp_fill = _default_fill_value
    try:
        humid_fill = humids._FillValue
    except AttributeError:
        humid_fill = _default_fill_value
    try:
        geo_fill = geopotential_heights._FillValue
    except AttributeError:
        geo_fill = _default_fill_value

    # Project lat/lon grid so it's regular (for easy interpolation)

    # See http://www.pkrc.net/wrf-lambert.html
    projection = pyproj.Proj(proj='lcc', lat_1=out.TRUELAT1,
                             lat_2=out.TRUELAT2, lat_0=out.MOAD_CEN_LAT,
                             lon_0=out.STAND_LON, a=6370, b=6370,
                             towgs84=(0,0,0), no_defs=True)

    lla = pyproj.Proj(proj='latlong')

    xs, ys = pyproj.transform(lla, projection, lons.flatten(), lats.flatten())
    xs = xs.reshape(lats.shape)
    ys = ys.reshape(lons.shape)

    # At this point, if all has gone well, xs has every column the same,
    # and ys has every row the same. Maybe they're not exactly the same
    # (due to rounding errors), so we'll average them.
    xs = np.mean(xs, axis=0)
    ys = np.mean(ys, axis=1)

    return reader.import_grids(xs=xs, ys=ys, pressure=plevs,
                               temperature=temps[0], temp_fill=temp_fill,
                               humidity=humids[0], humid_fill=humid_fill,
                               geo_ht=geopotential_heights[0],
                               geo_ht_fill=geo_fill, k1=_k1, k2=_k2, k3=_k3,
                               projection=projection,
                               scipy_interpolate=scipy_interpolate)


def load(out, plev, scipy_interpolate=False):
    """Load a NetCDF weather model as a NetCDFModel object."""
    with netcdf.netcdf_file(out) as f:
        with netcdf.netcdf_file(plev) as g:
            return _read_netcdf(f, g, scipy_interpolate)
