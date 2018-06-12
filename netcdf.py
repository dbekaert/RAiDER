"""Netcdf reader for weather data."""


# TODO: use a different method of interpolation (someone told me linear
# is inaccurate).


import scipy.interpolate as interpolate
import scipy.io.netcdf as netcdf


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
        temperature = self._t_inp(x, y, z)[0]
        relative_humidity = self._rh_inp(x, y, z)[0]
        # Calculate the saturation vapor pressure as in Hobiger et al.,
        # 2008. Maybe we want to do this like TRAIN instead.
        p_w = (10.79574*(1 - 273.16/T) - 5.028*numpy.log10(T/273.16)
                + 1.50475e-4*(1 - 10**(-8.2969*(T/273.16 - 1)))
                + 0.42873e-3*(10**(-4.76955*(1 - 273.16/T) - 1))
                + 0.78614444)
        # It's called p_v in Hobiger, but everyone else calls it e
        e = relative_humidity/100*p_w
        return k2*e/temperature + k3*e/temperature**2

    def point_hydrostatic_delay(self, x, y, z):
        """Calculate hydrostatic delay at a point."""
        P = numpy.exp(self._p_inp(x, y, z)[0])
        T = self._t_inp(x, y, z)[0]
        return k1*P/T # Hanssen, 2001


def toXYZ(lat, lon, ht):
    """Convert lat, lon, geopotential height to x, y, z in ECEF."""
    # This all comes straight from Wikipedia
    a = 6378137.0 # equatorial radius
    b = 6356752.3 # polar radius
    e2 = 1 - b**2/a**2 # square of first numerical eccentricity of the
                       # ellipsoid
    N = a/numpy.sqrt(1 - e2*numpy.sin(lat)**2)
    x = (N(lat) + ht)*numpy.cos(lat)*numpy.cos(lon)
    y = (N(lat) + ht)*numpy.cos(lat)*numpy.sin(lon)
    z = (b**2/a**2*N(lat) + ht)*numpy.sin(lat)
    return numpy.array((x, y, z))


def read_netcdf(out, plev):
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
    rows, cols = lats.shape
    def to1D(lvl, row, col):
        return lvl * rows * cols + row * cols + col
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
