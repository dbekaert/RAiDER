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
import numpy as np
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
    pass


def _common_delay(delay, lats, lons, heights, look_vecs, raytrace):
    """Perform computation common to hydrostatic and wet delay."""
    # Deal with Zenith special value, and non-raytracing method
    if raytrace:
        correction = None
    else:
        correction = 1/util.cosd(look_vecs)
        look_vecs = Zenith
    if look_vecs is Zenith:
        look_vecs = (np.array((util.cosd(lats)*util.cosd(lons),
                                  util.cosd(lats)*util.sind(lons),
                                  util.sind(lats))).T
                            * (_zref - heights).reshape(-1,1))

    lengths = np.linalg.norm(look_vecs, axis=-1)
    steps = np.array(np.ceil(lengths / _step), dtype=np.int64)
    indices = np.cumsum(steps)

    # We want the first index to be 0, and the others shifted
    indices = np.roll(indices, 1)
    indices[0] = 0

    start_positions = np.array(util.lla2ecef(lats, lons, heights)).T

    scaled_look_vecs = look_vecs / lengths.reshape(-1, 1)

    positions_l = list()
    t_points_l = list()
    # Please do it without a for loop
    for i in range(len(steps)):
        thisspace = np.linspace(0, lengths[i], steps[i])
        t_points_l.append(thisspace)
        position = start_positions[i] + thisspace.reshape(-1, 1) * scaled_look_vecs[i]
        positions_l.append(position)

    positions_a = np.concatenate(positions_l)

    wet_delays = delay(positions_a)

    delays = np.zeros(lats.shape[0])
    for i in range(len(steps)):
        start = indices[i]
        length = steps[i]
        chunk = wet_delays[start:start + length]
        t_points = t_points_l[i]
        delays[i] = 1e-6 * np.trapz(chunk, t_points)

    # Finally apply cosine correction if applicable
    if correction is not None:
        delays *= correction

    return delays


def wet_delay(weather, lats, lons, heights, look_vecs, raytrace=True):
    """Compute wet delay along the look vector."""
    return _common_delay(weather.wet_delay, lats, lons, heights, look_vecs,
                         raytrace)


def hydrostatic_delay(weather, lats, lons, heights, look_vecs, raytrace=True):
    """Compute hydrostatic delay along the look vector."""
    return _common_delay(weather.hydrostatic_delay, lats, lons, heights,
                         look_vecs, raytrace)


def delay_over_area(weather, lat_min, lat_max, lat_res, lon_min, lon_max,
                    lon_res, ht_min, ht_max, ht_res, los=Zenith):
    """Calculate (in parallel) the delays over an area."""
    lats = np.arange(lat_min, lat_max, lat_res)
    lons = np.arange(lon_min, lon_max, lon_res)
    hts = np.arange(ht_min, ht_max, ht_res)
    # It's the cartesian product (thanks StackOverflow)
    llas = np.array(np.meshgrid(lats, lons, hts)).T.reshape(-1,3)
    return delay_from_grid(weather, llas, los, parallel=True)


def _parallel_worker(job_type, start, end):
    """Calculate delay at a single index."""
    # please_cow contains the data we'd like to be CoW'd into the
    # subprocesses
    global please_cow
    weather, lats, lons, hts, los, raytrace = please_cow
    if los is Zenith:
        my_los = Zenith
    else:
        my_los = los[start:end]
    if job_type == 'hydro':
        return hydrostatic_delay(weather, lats[start:end], lons[start:end],
                                 hts[start:end], my_los, raytrace=raytrace)
    if job_type == 'wet':
        return wet_delay(weather, lats[start:end], lons[start:end],
                         hts[start:end], my_los, raytrace=raytrace)
    raise ValueError('Unknown job type {}'.format(job_type))


def delay_from_grid(weather, llas, los, parallel=False, raytrace=True):
    """Calculate delay on every point in a list.

    weather is the weather object, llas is a list of lat, lon, ht points
    at which to calculate delay, and los an array of line-of-sight
    vectors at each point. Pass parallel=True if you want to have real
    speed.
    """
    lats, lons, hts = llas.T

    # TRAIN rounds the DEM up to 0, so we will do so as well.
    hts[hts < 0] = 0

    if parallel:
        num_procs = os.cpu_count()

        hydro_procs = num_procs // 2
        wet_procs = num_procs - hydro_procs

        # Divide up jobs into an appropriate number of pieces
        hindices = np.linspace(0, len(llas), hydro_procs + 1, dtype=int)
        windices = np.linspace(0, len(llas), wet_procs + 1, dtype=int)

        # Store some things in global memory
        global please_cow
        please_cow = weather, lats, lons, hts, los, raytrace

        # Map over the jobs
        hjobs = (('hydro', hindices[i], hindices[i + 1])
                for i in range(hydro_procs))
        wjobs = (('wet', hindices[i], hindices[i + 1])
                for i in range(wet_procs))
        jobs = itertools.chain(hjobs, wjobs)
        with multiprocessing.Pool() as p:
            result = p.starmap(_parallel_worker, jobs)

        # Collect results
        hydro = np.concatenate(result[:hydro_procs])
        wet = np.concatenate(result[hydro_procs:])
    else:
        hydro = hydrostatic_delay(weather, lats, lons, hts, los,
                                  raytrace=raytrace)
        wet = wet_delay(weather, lats, lons, hts, los, raytrace=raytrace)
    return hydro, wet


def delay_from_files(weather, lat, lon, ht, parallel=False, los=Zenith,
                     raytrace=True):
    """Read location information from files and calculate delay."""
    lats = util.gdal_open(lat)
    lons = util.gdal_open(lon)
    hts = util.gdal_open(ht)

    if los is not Zenith:
        incidence, heading = util.gdal_open(los)
        los = util.los_to_lv(incidence, heading, lats, lons, hts).reshape(-1,3)

    # We need the three to be the same shape so that we know what to
    # reshape hydro and wet to. Plus, them being different sizes
    # indicates a definite user error.
    if not (lats.shape == lons.shape == hts.shape):
        raise ValueError(f'lat, lon, and ht should have the same shape, but '
                'instead lat had shape {lats.shape}, lon had shape '
                '{lons.shape}, and ht had shape {hts.shape}')

    llas = np.stack((lats.flatten(), lons.flatten(), hts.flatten()), axis=1)
    hydro, wet = delay_from_grid(weather, llas, los,
                                 parallel=parallel, raytrace=raytrace)
    hydro, wet = np.stack((hydro, wet)).reshape((2,) + lats.shape)
    return hydro, wet


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

    hts = np.zeroes((lats.size, lons.size)) # TODO: ???

    geo2rdrObj.set_geo_coordinate(lon_min, lat_min,
                                  lon_res, lat_res,
                                  hts)

    # compute the radar coordinate for each geo coordinate
    geo2rdrObj.geo2rdr()

    # get back the line of sight unit vector
    # TODO: should I really convert to an array?
    los = np.array(geo2rdrObj.get_los())

    # get back the slant ranges
    slant_range = geo2rdrObj.get_slant_range()

    los = slant_range * los

    return delay_over_area(weather, lat_min, lat_max, lat_res,
                           lon_min, lon_max, lon_res,
                           ht_min, ht_max, ht_res, los=los)


def los_to_lv(los):
    incidence = los[0]
    heading = los[1]
    xs = util.sind(incidence)*util.cosd(heading + 90)
    ys = util.sind(incidence)*util.sind(heading + 90)
    zs = util.cosd(incidence)
    return xs, ys, zs
