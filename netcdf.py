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
        # Calculate the saturation vapor pressure as in Hobiger et al.,
        # 2008. Maybe we want to do this like TRAIN instead.
        p_w = (10.79574*(1 - 273.16/T) - 5.028*numpy.log10(T/273.16)
                + 1.50475e-4*(1 - 10**(-8.2969*(T/273.16 - 1)))
                + 0.42873e-3*(10**(-4.76955*(1 - 273.16/T) - 1))
                + 0.78614444)
        # It's called p_v in Hobiger, but everyone else calls it e
        e = relative_humidity/100*p_w
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
    # I really hope these are always the same
    if lats.size != lons.size:
        raise NetCDFException
    outlength = lats.size * len(plevs)
    points = numpy.zeros((outlength, 3), dtype=lats.dtype)
    rows, cols = lats.shape
    def to1D(lvl, row, col):
        return lvl * rows * cols + row * cols + col
    # This one's a bit weird. temps and humids are easier.
    new_plevs = numpy.zeros(outlength)
    for lvl in range(len(plevs)):
        p = plevs[lvl]
        for row in range(rows):
            for col in range(cols):
                new_plevs[to1D(lvl, row, col)] = p
    # For temperature and humidity, we just make them flat.
    new_temps = numpy.reshape(temps, (outlength,))
    new_humids = numpy.reshape(humids, (outlength,))
    for lvl in range(len(plevs)):
        for row in range(rows):
            for col in range(cols):
                lat = lats[row][col]
                lon = lons[row][col]
                ht = geopotential_heights[lvl][row][col]
                points[to1D(lvl, row, col)] = _toXYZ(lat, lon, ht)
    # TODO: think about whether array copying is necessary
    return NetCDFModel(points=points, pressure=new_plevs,
                       temperature=new_temps, humidity=new_humids)


def load(out, plev):
    """Load a NetCDF weather model as a NetCDFModel object."""
    with netcdf.netcdf_file(out) as f:
        with netcdf.netcdf_file(plev) as g:
            return _read_netcdf(f, g)
