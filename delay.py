"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently we take samples every _step meters, which causes either
inaccuracies or inefficiencies, and we no longer can integrate to
infinity. We could develop a more advanced integrator to deal with these
issues, and probably I will. It goes pretty quickly right now, though.
"""


import demdownload
import era
from osgeo import gdal
gdal.UseExceptions()
import itertools
import losreader
import numpy as np
import os
import queue
import threading
import util
import wrf


# Step in meters to use when integrating
_step = 15

# Top of the troposphere
_zref = util.zref


class Zenith:
    """Special value indicating a look vector of "zenith"."""
    pass


def _too_high(positions, zref):
    positions_ecef = np.moveaxis(positions, -1, 0)
    positions_lla = np.stack(util.ecef2lla(*positions_ecef))
    high_indices = np.where(positions_lla[2] > zref)[0]
    first_high_index = high_indices[0] if high_indices.size else len(positions)
    return first_high_index


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
                            * (_zref - heights)[..., np.newaxis])

    lengths = np.linalg.norm(look_vecs, axis=-1)
    steps = np.array(np.ceil(lengths / _step), dtype=np.int64)
    start_positions = np.array(util.lla2ecef(lats, lons, heights)).T

    scaled_look_vecs = look_vecs / lengths[..., np.newaxis]

    positions_l = list()
    t_points_l = list()
    for i in range(len(steps)):
        thisspace = np.linspace(0, lengths[i], steps[i])
        position = (start_positions[i]
                + thisspace[..., np.newaxis]*scaled_look_vecs[i])
        first_high_index = _too_high(position, _zref)
        t_points_l.append(thisspace[:first_high_index])
        positions_l.append(position[:first_high_index])

        # Also correct the number of steps
        steps[i] = first_high_index

    positions_a = np.concatenate(positions_l)

    wet_delays = delay(positions_a)

    # Compute starting indices
    indices = np.cumsum(steps)
    # We want the first index to be 0, and the others shifted
    indices = np.roll(indices, 1)
    indices[0] = 0

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


def _parmap(f, i):
    """Execute f on elements of i in parallel."""
    # Queue of jobs
    q = queue.Queue()
    # Space for answers
    answers = list()
    for idx, x in enumerate(i):
        q.put((idx, x))
        answers.append(None)

    def go():
        while True:
            try:
                i, elem = q.get_nowait()
            except queue.Empty:
                break
            answers[i] = f(elem)

    threads = [threading.Thread(target=go) for _ in range(os.cpu_count())]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return answers


def delay_from_grid(weather, llas, los, parallel=False, raytrace=True):
    """Calculate delay on every point in a list.

    weather is the weather object, llas is a list of lat, lon, ht points
    at which to calculate delay, and los an array of line-of-sight
    vectors at each point. Pass parallel=True if you want to have real
    speed.
    """

    # Save the shape so we can restore later, but flatten to make it
    # easier to think about
    real_shape = llas.shape[:-1]
    llas = llas.reshape(-1, 3)
    # los can either be a bunch of vectors or a bunch of scalars. If
    # raytrace, then it's vectors, otherwise scalars. (Or it's Zenith)
    if los is not Zenith:
        if raytrace:
            los = los.reshape(-1, 3)
        else:
            los = los.flatten()

    lats, lons, hts = np.moveaxis(llas, -1, 0)

    if parallel:
        num_procs = os.cpu_count()

        hydro_procs = num_procs // 2
        wet_procs = num_procs - hydro_procs

        # Divide up jobs into an appropriate number of pieces
        hindices = np.linspace(0, len(llas), hydro_procs + 1, dtype=int)
        windices = np.linspace(0, len(llas), wet_procs + 1, dtype=int)

        # Build the jobs
        hjobs = (('hydro', hindices[i], hindices[i + 1])
                for i in range(hydro_procs))
        wjobs = (('wet', hindices[i], hindices[i + 1])
                for i in range(wet_procs))
        jobs = itertools.chain(hjobs, wjobs)

        # Parallel worker
        def go(job):
            job_type, start, end = job
            if los is Zenith:
                my_los = Zenith
            else:
                my_los = los[start:end]
            if job_type == 'hydro':
                return hydrostatic_delay(weather, lats[start:end],
                                         lons[start:end], hts[start:end],
                                         my_los, raytrace=raytrace)
            if job_type == 'wet':
                return wet_delay(weather, lats[start:end], lons[start:end],
                                 hts[start:end], my_los, raytrace=raytrace)
            raise ValueError('Unknown job type {}'.format(job_type))

        # Execute the parallel worker
        result = _parmap(go, jobs)

        # Collect results
        hydro = np.concatenate(result[:hydro_procs])
        wet = np.concatenate(result[hydro_procs:])
    else:
        hydro = hydrostatic_delay(weather, lats, lons, hts, los,
                                  raytrace=raytrace)
        wet = wet_delay(weather, lats, lons, hts, los, raytrace=raytrace)

    # Restore shape
    hydro, wet = np.stack((hydro, wet)).reshape((2,) + real_shape)

    return hydro, wet


def delay_from_files(weather, lat, lon, ht, parallel=False, los=Zenith,
                     raytrace=True):
    """Read location information from files and calculate delay."""
    lats = util.gdal_open(lat)
    lons = util.gdal_open(lon)
    hts = util.gdal_open(ht)

    if los is not Zenith:
        incidence, heading = util.gdal_open(los)
        if raytrace:
            los = util.los_to_lv(
                    incidence, heading, lats, lons, hts).reshape(-1,3)
        else:
            los = incidence.flatten()

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
                t, x, y, z, vx, vy, vz, hts):
    """Calculate delay over an area using state vectors.

    The information about the sensor is given by t, x, y, z, vx, vy, vz.
    Other parameters specify the region of interest. The returned object
    will be hydrostatic and wet arrays covering the indicated area.
    """

    los = np.stack(util.state_to_los(t, x, y, z, vx, vy, vz, lon_min, lon_res,
                                     lat_min, lat_res, hts), axis=-1)

    latlin = np.linspace(lat_min, lat_max, (lat_max - lat_min)/lat_res)
    lonlin = np.linspace(lon_min, lon_max, (lon_max - lon_min)/lon_res)

    lons, lats = np.meshgrid(lonlin, latlin)

    llas = np.stack((lats, lons, hts), axis=-1)

    return delay_from_grid(weather, llas, los, parallel=True)


def _tropo_delay_with_values(los, lats, lons, hts, weather, zref, out, time):
    # LOS
    if los is None:
        los = Zenith
    else:
        los = losreader.infer_los(los, lats, lons, hts, zref)

    # We want to test if any shapes are different
    if (not hts.shape == lats.shape == lons.shape
            or los is not Zenith and los.shape[:-1] != hts.shape):
        raise ValueError('I need lats, lons, heights, and los to all be the '
            f'same shape. lats had shape {lats.shape}, lons had shape '
            f'{lons.shape}, heights had shape {hts.shape}, and los had shape '
            f'{los.shape}')

    # Do the calculation
    llas = np.stack((lats, lons, hts), axis=-1)
    hydro, wet = delay_from_grid(weather, llas, los, parallel=True,
                                 raytrace=True)
    return hydro, wet


def tropo_delay(los, lat, lon, heights, weather, zref, out, time):
    weather_fmt, weather_type, weather_files = weather

    # Lat, lon
    if lat is None:
        # They'll get set later with weather
        lats = lons = None
    else:
        lats = util.gdal_open(lat)
        lons = util.gdal_open(lon)

    # Height
    height_type, height_info = heights
    if height_type == 'dem':
        hts = util.gdal_open(height_info)
    elif height_type == 'lvs':
        hts = height_info
    elif height_type == 'download':
        hts = demdownload.download_dem(lats, lons)
    else:
        raise ValueError(f'Unexpected height_type {repr(height_type)}')

    # Make weather
    if weather_fmt == 'wrf':
        if weather_type == 'files':
            weather = wrf.load(*weather_files, zmin=np.min(hts))

            # Let lats and lons to weather model nodes if necessary
            if lats is None:
                lats, lons = wrf.wm_nodes(*weather_files)
        else:
            raise ValueError('weather_type should be files with wrf model, '
                f'got {repr(weather_type)}')
    if weather_fmt in ('era-i', 'era5'):
        if weather_type == 'netcdf':
            weather = era.load(*weather_files)
            if lats is None:
                lats, lons = era.wm_nodes(*weather_files)
        elif weather_type == 'download':
            if lats is None:
                raise ValueError(
                        "Can't use delay at weather model nodes if you also "
                            "want me to download the weather model")
            if weather_fmt == 'era-i':
                weather = era.fetch_era_interim(lats, lons, time)
            else:
                weather = era.fetch_era5(lats, lons, time)
        else:
            raise ValueError(
                    'weather_type should be download or netcdf, but it was '
                    f'{repr(weather_type)}')
    else:
        raise ValueError(f'Unknown weather model type {repr(weather_fmt)}')

    # For later
    drv = gdal.GetDriverByName('ENVI')
    hydroname, wetname = (f'{weather_fmt}_{dtyp}_'
        f'{time.isoformat() + "_" if time is not None else ""}delay.envi'
            for dtyp in ('hydro', 'wet'))

    # Pretty different calculation depending on whether they specified a
    # list of heights or just a DEM
    if height_type == 'lvs':
        shape = (len(hts),) + lats.shape
        total_hydro = np.zeros(shape)
        total_wet = np.zeros(shape)
        for i, ht in enumerate(hts):
            hydro, wet = _tropo_delay_with_values(
                    los, lats, lons, np.broadcast_to(ht, lats.shape), weather,
                    zref, out, time)
            total_hydro[i] = hydro
            total_wet[i] = wet

        hydro_ds = drv.Create(
                os.path.join(out, hydroname), total_hydro.shape[2],
                total_hydro.shape[1], len(hts), gdal.GDT_Float64)
        for lvl, (hydro, ht) in enumerate(zip(total_hydro, hts), start=1):
            band = hydro_ds.GetRasterBand(lvl)
            band.SetDescription(str(ht))
            band.WriteArray(hydro)
        hydro_ds = None

        wet_ds = drv.Create(
                os.path.join(out, wetname), total_wet.shape[2],
                total_wet.shape[1], len(hts), gdal.GDT_Float64)
        for lvl, (wet, ht) in enumerate(zip(total_wet, hts), start=1):
            band = wet_ds.GetRasterBand(lvl)
            band.SetDescription(str(ht))
            band.WriteArray(wet)
        wet_ds = None

    else:
        hydro, wet = _tropo_delay_with_values(
                los, lats, lons, hts, weather, zref, out, time)

        # Write the output file
        # TODO: maybe support other files than ENVI
        hydro_ds = drv.Create(
                os.path.join(out, hydroname), hydro.shape[1], hydro.shape[0],
                1, gdal.GDT_Float64)
        hydro_ds.GetRasterBand(1).WriteArray(hydro)
        hydro_ds = None
        wet_ds = drv.Create(
                os.path.join(out, wetname), wet.shape[1], wet.shape[0], 1,
                gdal.GDT_Float64)
        wet_ds.GetRasterBand(1).WriteArray(wet)
        wet_ds = None
