"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently we take samples every _step meters, which causes either
inaccuracies or inefficiencies, and we no longer can integrate to
infinity. We could develop a more advanced integrator to deal with these
issues, and probably I will. It goes pretty quickly right now, though.
"""


from osgeo import gdal
gdal.UseExceptions()
import itertools
import multiprocessing
import numpy
import os
import progressbar
import util
try:
    import Geo2rdr
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


def _common_delay(delay, lats, lons, heights, look_vecs):
    """Perform computation common to hydrostatic and wet delay."""
    lengths = numpy.linalg.norm(look_vecs, axis=-1)
    steps = numpy.array(numpy.ceil(lengths / _step), dtype=numpy.int64)
    indices = numpy.cumsum(steps)

    # We want the first index to be 0, and the others shifted
    indices = numpy.roll(indices, 1)
    indices[0] = 0

    start_positions = numpy.array(util.lla2ecef(lats, lons, heights)).T

    scaled_look_vecs = look_vecs / lengths.reshape(-1, 1)

    positions_l = list()
    t_points_l = list()
    # Please do it without a for loop
    for i in range(len(steps)):
        thisspace = numpy.linspace(0, lengths[i], steps[i])
        t_points_l.append(thisspace)
        position = start_positions[i] + thisspace.reshape(-1, 1) * scaled_look_vecs[i]
        positions_l.append(position)

    positions_a = numpy.concatenate(positions_l)

    wet_delays = delay(positions_a)

    delays = numpy.zeros(lats.shape[0])
    for i in range(len(steps)):
        start = indices[i]
        length = steps[i]
        chunk = wet_delays[start:start + length]
        t_points = t_points_l[i]
        delays[i] = 1e-6 * numpy.trapz(chunk, t_points)

    return delays


def wet_delay(weather, lats, lons, heights, look_vecs):
    """Compute wet delay along the look vector."""
    return _common_delay(weather.wet_delay, lats, lons, heights, look_vecs)


def hydrostatic_delay(weather, lats, lons, heights, look_vecs):
    """Compute hydrostatic delay along the look vector."""
    return _common_delay(weather.hydrostatic_delay, lats, lons, heights,
                         look_vecs)


def delay_over_area(weather, lat_min, lat_max, lat_res, lon_min, lon_max,
                    lon_res, ht_min, ht_max, ht_res, los=Zenith()):
    """Calculate (in parallel) the delays over an area."""
    lats = numpy.arange(lat_min, lat_max, lat_res)
    lons = numpy.arange(lon_min, lon_max, lon_res)
    hts = numpy.arange(ht_min, ht_max, ht_res)
    # It's the cartesian product (thanks StackOverflow)
    llas = numpy.array(numpy.meshgrid(lats, lons, hts)).T.reshape(-1,3)
    return delay_from_grid(weather, llas, los, parallel=True)


def _delay_from_grid_work(weather, llas, los, raytrace, i):
    """Worker function for integrating delay.

    This can't be called directly within multiprocessing since we don't
    want weather, llas, and los to be copied. But this function is
    what does the real work."""
    lat, lon, ht = llas[i]
    if not isinstance(los, Zenith):
        position = numpy.array(util.lla2ecef(lat, lon, ht))
        look_vec = los[i]
        if not raytrace:
            pos = look_vec
            look_vec = Zenith(_zref)
            correction = numpy.linalg.norm(pos) / _zref
        else:
            correction = 1
    else:
        look_vec = los
        correction = 1
    hydro = correction * hydrostatic_delay(weather, lat, lon, ht, look_vec)
    wet = correction * wet_delay(weather, lat, lon, ht, look_vec)
    return hydro, wet


def _parallel_worker(hydro, start, end):
    """Calculate delay at a single index."""
    # please_cow contains the data we'd like to be CoW'd into the
    # subprocesses
    global please_cow
    weather, lats, lons, hts, los, raytrace = please_cow
    if hydro:
        return hydrostatic_delay(weather, lats[start:end], lons[start:end],
                                 hts[start:end], los[start:end])
    return wet_delay(weather, lats[start:end], lons[start:end], hts[start:end],
                     los[start:end])


def delay_from_grid(weather, llas, los, parallel=False, raytrace=True):
    """Calculate delay on every point in a list.

    weather is the weather object, llas is a list of lat, lon, ht points
    at which to calculate delay, and los an array of line-of-sight
    vectors at each point. Pass parallel=True if you want to have real
    speed.
    """
    lats, lons, hts = llas.T
    if los is Zenith:
        los = numpy.array((util.cosd(lats)*util.cosd(lons),
            util.cosd(lats)*util.sind(lons), util.sind(lats))).T * (_zref - llas[:,2]).reshape(-1,1)
    if parallel:
        num_procs = os.cpu_count()

        hydro_procs = num_procs // 2
        wet_procs = num_procs - hydro_procs

        # Divide up jobs into an appropriate number of pieces
        hindices = numpy.linspace(0, len(llas), hydro_procs + 1, dtype=int)
        windices = numpy.linspace(0, len(llas), wet_procs + 1, dtype=int)

        # Store some things in global memory
        global please_cow
        please_cow = weather, lats, lons, hts, los, raytrace

        # Map over the jobs
        # TODO: magic true and false
        hjobs = ((True, hindices[i], hindices[i + 1]) for i in range(hydro_procs))
        wjobs = ((False, hindices[i], hindices[i + 1]) for i in range(wet_procs))
        jobs = itertools.chain(hjobs, wjobs)
        with multiprocessing.pool.Pool() as p:
            result = p.starmap(_parallel_worker, jobs)

        # Collect results
        hydro = numpy.concatenate(result[:hydro_procs])
        wet = numpy.concatenate(result[hydro_procs:])
    else:
        hydro = hydrostatic_delay(weather, lats, lons, hts, los)
        wet = wet_delay(weather, lats, lons, hts, los)
    return hydro, wet


def delay_from_files(weather, lat, lon, ht, parallel=False, los=Zenith(),
                     raytrace=True):
    """Read location information from files and calculate delay."""
    lats_file = gdal.Open(lat)
    lats = lats_file.ReadAsArray()
    del lats_file
    lons_file = gdal.Open(lon)
    lons = lons_file.ReadAsArray()
    del lons_file

    hts = gdal.Open(ht).ReadAsArray()
    llas = numpy.stack((lats.flatten(), lons.flatten(), hts.flatten()), axis=1)
    return delay_from_grid(weather, llas, los, parallel=parallel,
                           raytrace=raytrace)


def slant_delay(weather, lat_min, lat_max, lat_res, lon_min, lon_max, lon_res,
                ht_min, ht_max, ht_step, t, x, y, z, vx, vy, vz):
    """Calculate delay over an area using state vectors.

    The information about the sensor is given by t, x, y, z, vx, vy, vz.
    Other parameters specify the region of interest. The returned object
    will be hydrostatic and wet arrays covering the indicated area.
    """
    for i, st in enumerate(obj.orbit.stateVectors):
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
    # TODO: should I really convert to an array?
    los = numpy.array(geo2rdrObj.get_los())

    # get back the slant ranges
    slant_range = geo2rdrObj.get_slant_range()

    los = slant_range * los

    return delay_over_area(weather, lat_min, lat_max, lat_res,
                           lon_min, lon_max, lon_res,
                           ht_min, ht_max, ht_res, los=los)


def los_from_position(lats, lons, hts, sensor):
    """In this case, sensor is lla, but it could easily be xyz."""
    sensorx, sensory, sensorz = util.lla2ecef(*sensor)
    xs, ys, zs = util.lla2ecef(lats, lons, hts)
    return xs - sensorx, ys - sensory, zs - sensorz
