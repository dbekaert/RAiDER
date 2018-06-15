"""Netcdf reader for weather data."""


# TODO: use a different method of interpolation (someone told me linear
# is inaccurate).
# TODO: (maybe) add another layer below to make sure we get everything


import numpy
import scipy.interpolate as interpolate
import scipy.io.netcdf as netcdf
import util


class NetCDFException(Exception):
    """Exception for unexpected values from the NetCDF file we read."""
    pass


class NetCDFModel:
    """Weather model for NetCDF. Knows dry and hydrostatic delay."""
    def __init__(self, points, pressure, temperature, humidity):
        """Initialize a NetCDFModel."""
        # Log of pressure since pressure is interpolated exponentially
        self._p_inp = interpolate.LinearNDInterpolator(points,
                                                       numpy.log(pressure))
        self._t_inp = interpolate.LinearNDInterpolator(points, temperature)
        self._rh_inp = interpolate.LinearNDInterpolator(points, humidity)

    def pressure(self, x, y, z):
        """Calculate pressure at a point."""
        return numpy.exp(self._p_inp(x, y, z))

    def temperature(self, x, y, z):
        """Calculate temperature at a point."""
        return self._t_inp(x, y, z)

    def rel_humid(self, x, y, z):
        """Calculate relative humidity at a point."""
        return self._rh_inp(x, y, z)

    # Parameters from Hanssen, 2001
    k1 = 0.776 # [K/Pa]
    # Should be k2'
    k2 = 0.233 # [K/Pa]
    k3 = 3.75e3 # [K^2/Pa]


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


def _propagate_down(a):
    """Pull real values down to cover NaNs

    a might contain some NaNs which live under real values. We replace
    those NaNs with actual values. a must be a 3D array.
    """
    out = a.copy()
    z, x, y = out.shape
    for i in range(x):
        for j in range(y):
            held = None
            for k in range(z - 1, -1, -1):
                val = out[k][i][j]
                if numpy.isnan(val) and held is not None:
                    out[k][i][j] = held
                elif not numpy.isnan(val):
                    held = val
    return out


def _read_netcdf(out, plev):
    """Return a NetCDFModel given open netcdf files."""
    # n.b.: all of these things we read are arrays of length 1, so
    # we get the first element to access the actual data.
    lats = out.variables['XLAT'][0]
    # Why is it XLONG with a G? Maybe the G means geo (but then why
    # isn't it XLATG?).
    lons = out.variables['XLONG'][0]

    if plev.variables['P_PL'].units.decode('utf-8') == 'Pa':
        plevs = plev.variables['P_PL'][0]
    else:
        err = "Unknown units for pressure: '{}'"
        raise NetCDFException(err.format(plev.variables['P_PL'].units))

    if plev.variables['T_PL'].units.decode('utf-8') == 'K':
        temps = plev.variables['T_PL'][0]
    else:
        err = "Unknown units for temperature: '{}'"
        raise NetCDFException(err.format(plev.variables['T_PL'].units))

    # TODO: extract partial pressure directly (q?)
    if plev.variables['RH_PL'].units.decode('utf-8') == '%':
        humids = plev.variables['RH_PL'][0]
    else:
        err = "Unknown units for relative humidity: '{}'"
        raise NetCDFException(err.format(plev.variables['RH_PL'].units))

    if plev.variables['GHT_PL'].units.decode('utf-8') == 'm':
        geopotential_heights = plev.variables['GHT_PL'][0]
    else:
        err = "Unknown units for geopotential height: '{}'"
        raise NetCDFException(err.format(plev.variables['GHT_PL'].units))

    # Replacing the non-useful values by NaN, and fill in values under
    # the topography
    temps_fixed = _propagate_down(numpy.where(temps != -999, temps, numpy.nan))
    humids_fixed = _propagate_down(numpy.where(humids != -999, humids,
                                               numpy.nan))
    geo_ht_fix = _propagate_down(numpy.where(geopotential_heights != -999,
                                             geopotential_heights, numpy.nan))

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

    # So a while ago we removed all the NaNs under the earth, but there
    # are still going to be some left, so we just remove those points.
    points_thing = numpy.zeros(new_plevs.shape, dtype=bool)
    for i in range(new_plevs.size):
        points_thing[i] = numpy.all(numpy.logical_not(numpy.isnan(points[i])))
    ok = _big_and(numpy.logical_not(numpy.isnan(new_plevs)),
                  numpy.logical_not(numpy.isnan(new_temps)),
                  numpy.logical_not(numpy.isnan(new_humids)), points_thing)

    return points[ok], new_plevs[ok], new_temps[ok], new_humids[ok]


def load(out, plev):
    """Load a NetCDF weather model as a NetCDFModel object."""
    with netcdf.netcdf_file(out) as f:
        with netcdf.netcdf_file(plev) as g:
            points, plevs, temps, humids = _read_netcdf(f, g)
            return NetCDFModel(points=points, pressure=plevs,
                               temperature=temps, humidity=humids)
