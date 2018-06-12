"""Netcdf reader for weather data."""


import scipy.interpolate as interpolate
import scipy.io.netcdf as netcdf


class NetCDFException(Exception):
    pass


class NetCDFModel:
    def __init__(self, points, pressure, temperature, humidity):
        """Initialize a NetCDFModel."""
        # self._points = points
        # self._pressure = pressure
        # self._temperature = temperature
        # self._humidity = humidity
        self._p_inp = interpolate.LinearNDInterpolator(points, pressure)
        self._t_inp = interpolate.LinearNDInterpolator(points, temperature)
        self._e_inp = interpolate.LinearNDInterpolator(points, humidity)

    def point_dry_delay(self, x, y, z):
        """Calculate dry delay at a point."""
        # TODO: not implemented
        pass

    def point_hydrostatic_delay(self, x, y, z):
        """Calculate hydrostatic delay at a point."""
        # TODO: not implemented
        pass


def _toXYZ(lat, lon, ht):
    """Convert lat, lon, geopotential height to x, y, z in WGS84."""
    # TODO: not implemented
    # author's note: this is a bit complicated
    pass


def _read_netcdf(out, plev):
    """Return a NetCDFModel given open netcdf files."""
    # n.b.: all of these things we read are arrays of length 1, so
    # we get the first element to access the actual data.
    lats = out.variables['XLAT'][0]
    # Why is it XLONG with a G? Maybe the G means geo (but then why
    # isn't it XLATG?).
    lons = out.variables['XLONG'][0]
    plevs = plev.variables['P_PL'][0]
    temps = plev.variables['T_PL'][0]
    humids = plev.variables['RH_PL'][0]
    geopotential_heights = plev.variables['GHT_PL'][0]
    # I really hope these are always the same
    if lats.size != lons.size:
        raise NetCDFException
    points = numpy.zeros((lats.size, 3), dtype=lats.dtype)
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


def load(out, plev):
    """Load a NetCDF weather model as a NetCDFModel object."""
    with netcdf.netcdf_file(out) as f:
        with netcdf.netcdf_file(plev) as g:
            return _read_netcdf(f, g)
