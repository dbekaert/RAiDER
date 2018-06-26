"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently we take samples every _step meters, which causes either
inaccuracies or inefficiencies, and we no longer can integrate to
infinity. We could develop a more advanced integrator to deal with these
issues, and probably I will. It goes pretty quickly right now, though.
"""


from osgeo import gdal
import multiprocessing
import numpy
import progressbar
import util
try:
    import Geo2rdr
    from iscesys.Component.ProductManager import ProductManager as PM
except ImportError: # TODO: consider later
    pass


# Step in meters to use when integrating
_step = 15

# Top of the troposphere
_zref = 15000


class Zenith:
    """Special value indicating a look vector of "zenith"."""
    def __init__(self, rnge=None):
        self.rnge = rnge


def _common_delay(weather, lat, lon, height, look_vec):
    """Perform computation common to hydrostatic and wet delay."""
    position = util.lla2ecef(lat, lon, height)
    if isinstance(look_vec, Zenith):
        rnge = look_vec.rnge
        if rnge is None:
            rnge = _zref - height
        look_vec = numpy.array((util.cosd(lat)*util.cosd(lon),
                                util.cosd(lat)*util.sind(lon), util.sind(lat)))
        l = numpy.linalg.norm(look_vec)
        look_vec /= l
        look_vec *= rnge

    rnge = numpy.linalg.norm(look_vec)
    look_vec /= numpy.linalg.norm(look_vec)

    t_points = numpy.linspace(0, rnge, rnge / _step)

    wheres = position + look_vec * t_points.reshape((t_points.size,1))

    return t_points, wheres


def wet_delay(weather, lat, lon, height, look_vec):
    """Compute wet delay along the look vector."""
    t_points, wheres = _common_delay(weather, lat, lon, height, look_vec)

    wet_delays = weather.wet_delay(wheres)

    return 1e-6 * numpy.trapz(wet_delays, t_points)


def hydrostatic_delay(weather, lat, lon, height, look_vec):
    """Compute hydrostatic delay along the look vector."""
    t_points, wheres = _common_delay(weather, lat, lon, height, look_vec)

    delay = weather.hydrostatic_delay(wheres)

    return 1e-6 * numpy.trapz(delay, t_points)


def delay_over_area(weather, lat_min, lat_max, lat_res, lon_min, lon_max,
                    lon_res, ht_min, ht_max, ht_res, sensor=Zenith()):
    """Calculate (in parallel) the delays over an area."""
    lats = numpy.arange(lat_min, lat_max, lat_res)
    lons = numpy.arange(lon_min, lon_max, lon_res)
    hts = numpy.arange(ht_min, ht_max, ht_res)
    # It's the cartesian product (thanks StackOverflow)
    llas = numpy.array(numpy.meshgrid(lats, lons, hts)).T.reshape(-1,3)
    return delay_from_grid(weather, llas, sensor, parallel=True)


def _delay_from_grid_work(weather, llas, sensor, i):
    """Worker function for integrating delay.

    This can't be called directly within multiprocessing since we don't
    want weather, llas, and sensor to be copied. But this function is
    what does the real work."""
    lat, lon, ht = llas[i]
    if not isinstance(sensor, Zenith):
        position = util.lla2ecef(lat, lon, ht)
        look_vec = sensor - position
    else:
        look_vec = sensor
    hydro = hydrostatic_delay(weather, lat, lon, ht, look_vec)
    wet = wet_delay(weather, lat, lon, ht, look_vec)
    return hydro, wet


def _parallel_worker(i):
    """Calculate delay at a single index."""
    # please_cow contains the data we'd like to be CoW'd into the
    # subprocesses
    global please_cow
    weather, llas, sensor = please_cow
    return _delay_from_grid_work(weather, llas, sensor, i)


def delay_from_grid(weather, llas, sensor, parallel=False):
    """Calculate delay on every point in a list.

    weather is the weather object, llas is a list of lat, lon, ht points
    at which to calculate delay, and sensor is the location of the
    sensor. Pass parallel=True if you want to have real speed.
    """
    out = numpy.zeros((llas.shape[0], 2))
    if parallel:
        global please_cow
        please_cow = weather, llas, sensor
        with multiprocessing.Pool() as p:
            answers = p.map(_parallel_worker, range(llas.shape[0]))
    else:
        answers = (_delay_from_grid_work(weather, llas, sensor, i)
                for i in range(llas.shape[0]))
    for i, result in enumerate(answers):
        hydro_delay, wet_delay = result
        out[i][:] = (hydro_delay, wet_delay)
    return out


def delay_from_files(weather, lat, lon, ht, parallel=False, sensor=Zenith()):
    """Read location information from files and calculate delay."""
    lats = gdal.Open(lat).ReadAsArray()
    lons = gdal.Open(lon).ReadAsArray()
    hts = gdal.Open(ht).ReadAsArray()
    llas = numpy.stack((lats.flatten(), lons.flatten(), hts.flatten()), axis=1)
    return delay_from_grid(weather, llas, sensor, parallel=parallel)


def slant_delay(weather, lat_min, lat_max, lat_res, lon_min, lon_max, lon_res,
                ht_min, ht_max, ht_step, orbit):
    pm = PM()
    pm.configure()
    obj = pm.loadProduct(orbit)

    numSV = len(obj.orbit.stateVectors)
    t = numpy.ones(numSV)
    x = numpy.ones(numSV)
    y = numpy.ones(numSV)
    z = numpy.ones(numSV)
    vx = numpy.ones(numSV)
    vy = numpy.ones(numSV)
    vz = numpy.ones(numSV)

    for i,st in enumerate(obj.orbit.stateVectors):
        #tt = st.time
        #t[i] = datetime2year(tt)
        t[i] = st.time.second + st.time.minute*60.0
        x[i] = st.position[0]
        y[i] = st.position[1]
        z[i] = st.position[2]
        vx[i] = st.velocity[0]
        vy[i] = st.velocity[1]
        vz[i] = st.velocity[2]

    ###########################
    #Instantiate Geo2rdr

    geo2rdrObj = Geo2rdr.PyGeo2rdr()

    # pass the 1D arrays of state vectors to geo2rdr object
    geo2rdrObj.set_orbit(t, x, y, z, vx , vy , vz)

    # set the geo coordinates: lat and lon of the start pixel,
    #                           lat and lon steps
    #                           DEM heights

    hts = numpy.zeroes((lats.size, lons.size)) # TODO: ???

    geo2rdrObj.set_geo_coordinate(lon_min, lat_min,
                                  lon_res, lat_res,
                                  hts)

    # compute the radar coordinate for each geo coordinate
    geo2rdrObj.geo2rdr()

    # get back the line of sight unit vector
    los = numpy.array(geo2rdrObj.get_los())

    # get back the slant ranges
    slant_range = geo2rdrObj.get_slant_range()

    sensor = util.lla2ecef(lat_min, lon_min, ht_min) + slant_range * los

    return delay_over_area(weather, lat_min, lat_max, lat_res,
                           lon_min, lon_max, lon_res,
                           ht_min, ht_max, ht_res, sensor=sensor)
