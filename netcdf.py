"""Netcdf reader for weather data."""


# TODO: use a different method of interpolation (someone told me linear
# is inaccurate).
# TODO: preprocess file to remove -999s, and replace with above values
# TODO: (maybe) add another layer below to make sure we get everything


import numpy
import scipy.interpolate as interpolate
import scipy.io.netcdf as netcdf
import util


# Parameters from Hanssen, 2001
k1 = 0.776 # [K/Pa]
# Should be k2'
k2 = 0.233 # [K/Pa]
k3 = 3.75e3 # [K^2/Pa]


class NetCDFException(Exception):
    """Exception for unexpected values from the NetCDF file we read."""
    pass


def _calculate_e(temp, hum):
    """Calculate e, the partial pressure of water vapor.

    We're given Temperature and Humiditiy. Right now we're using the
    equations from TRAIN, even though they're different from Hobiger et
    al., 2008.
    """
    svpw = 6.1121*numpy.exp((17.502*(temp - 273.16))/(240.97 + temp - 273.16))
    svpi = 6.1121*numpy.exp((22.587*(temp - 273.16))/(273.86 + temp - 273.16))
    tempbound1 = 273.16 # 0
    tempbound2 = 250.16 # -23

    svp = svpw
    wgt = (temp - tempbound2)/(tempbound1 - tempbound2)
    svp = svpi + (svpw - svpi)*wgt**2
    if temp > tempbound1:
        return hum/100 * svpw
    elif temp < tempbound2:
        return hum/100 * svpi
    return hum/100 * svp


class NetCDFModel:
    """Weather model for NetCDF. Knows dry and hydrostatic delay."""
    def __init__(self, points, pressure, temperature, humidity):
        """Initialize a NetCDFModel."""
        # Log of pressure since pressure is interpolated exponentially
        self._p_inp = interpolate.LinearNDInterpolator(points,
                                                       numpy.log(pressure))
        self._t_inp = interpolate.LinearNDInterpolator(points, temperature)
        self._rh_inp = interpolate.LinearNDInterpolator(points, humidity)

    def point_dry_delay(self, x, y, z):
        """Calculate dry delay at a point."""
        T = self._t_inp(x, y, z)
        relative_humidity = self._rh_inp(x, y, z)
        e = _calculate_e(T, relative_humidity)
        delay = k2*e/T + k3*e/T**2
        return delay if not numpy.isnan(delay) else 0

    def point_hydrostatic_delay(self, x, y, z):
        """Calculate hydrostatic delay at a point."""
        pressure = numpy.exp(self._p_inp(x, y, z))
        temperature = self._t_inp(x, y, z)
        delay = k1*pressure/temperature # Hanssen, 2001
        return delay if not numpy.isnan(delay) else 0


def _toXYZ(lat, lon, ht):
    """Convert lat, lon, geopotential height to x, y, z in ECEF."""
    # Convert geopotential to geometric height. This comes straight from
    # TRAIN
    g0 = 9.80665
    # Map of g with latitude (I'm skeptical of this equation)
    g = 9.80616*(1 - 0.002637*util.cosd(2*lat)
            + 0.0000059*(util.cosd(2*lat))**2)
    Rmax = 6378137
    Rmin = 6356752
    Re = numpy.sqrt(1/(((util.cosd(lat)**2)/Rmax**2)
        + ((util.sind(lat)**2)/Rmin**2)))

    # Calculate Geometric Height, h
    h = (ht*Re)/(g/g0*Re - ht)
    return util.lla2ecef(lat, lon, h)


def _big_and(*args):
    result = args[0]
    for a in args[1:]:
        result = numpy.logical_and(result, a)
    return result


def _read_netcdf(out, plev):
    """Return a NetCDFModel given open netcdf files."""
    # n.b.: all of these things we read are arrays of length 1, so
    # we get the first element to access the actual data.
    lats = out.variables['XLAT'][0]
    # Why is it XLONG with a G? Maybe the G means geo (but then why
    # isn't it XLATG?).
    lons = out.variables['XLONG'][0]
    plevs = plev.variables['P_PL'][0] / 100 # Convert hPa to Pa
    temps = plev.variables['T_PL'][0]
    humids = plev.variables['RH_PL'][0]
    geopotential_heights = plev.variables['GHT_PL'][0]

    # Replacing the non-useful values by NaN
    temps_fixed = numpy.where(temps != -999, temps, numpy.nan)
    humids_fixed = numpy.where(humids != -999, humids, numpy.nan)
    geo_ht_fix = numpy.where(geopotential_heights != -999,
                             geopotential_heights, numpy.nan)

    # I really hope these are always the same
    if lats.size != lons.size:
        raise NetCDFException
    outlength = lats.size * len(plevs)
    points = numpy.zeros((outlength, 3), dtype=lats.dtype)
    rows, cols = lats.shape
    def to1D(lvl, row, col):
        return lvl * rows * cols + row * cols + col
    # This one's a bit weird. temps and humids are easier.
    new_plevs = numpy.zeros(outlength, dtype=plevs.dtype)
    for lvl in range(len(plevs)):
        p = plevs[lvl]
        for row in range(rows):
            for col in range(cols):
                new_plevs[to1D(lvl, row, col)] = p
    new_temps = numpy.reshape(temps_fixed, (outlength,))
    new_humids = numpy.reshape(humids_fixed, (outlength,))
    for lvl in range(len(plevs)):
        for row in range(rows):
            for col in range(cols):
                lat = lats[row][col]
                lon = lons[row][col]
                geo_ht = geo_ht_fix[lvl][row][col]
                pt_idx = to1D(lvl, row, col)
                points[pt_idx] = _toXYZ(lat, lon, geo_ht)

    # The issue now arises that some of the values are NaN. That's not
    # ok, so we go through the arduous process of removing those
    # elements.
    points_thing = numpy.zeros(new_plevs.shape, dtype=bool)
    for i in range(new_plevs.size):
        points_thing[i] = numpy.all(numpy.logical_not(numpy.isnan(points[i])))
    ok = _big_and(numpy.logical_not(numpy.isnan(new_plevs)),
                  numpy.logical_not(numpy.isnan(new_temps)),
                  numpy.logical_not(numpy.isnan(new_humids)), points_thing)
    num_ok = ok.size
    points_fix = points[ok]
    plevs_fix = new_plevs[ok]
    temps_fix = new_temps[ok]
    humids_fix = new_humids[ok]

    return points_fix, plevs_fix, temps_fix, humids_fix


def load(out, plev):
    """Load a NetCDF weather model as a NetCDFModel object."""
    with netcdf.netcdf_file(out) as f:
        with netcdf.netcdf_file(plev) as g:
            points, plevs, temps, humids = _read_netcdf(f, g)
            return NetCDFModel(points=points, pressure=plevs,
                               temperature=temps, humidity=humids)
