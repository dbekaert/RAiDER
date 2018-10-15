"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently we take samples every _STEP meters, which causes either
inaccuracies or inefficiencies, and we no longer can integrate to
infinity.
"""


from osgeo import gdal
gdal.UseExceptions()

# standard imports
import itertools
import numpy as np
import os
import pyproj
import tempfile
import queue
import threading

# local imports
import constants as const
import demdownload
import losreader
import util


# Step in meters to use when integrating
_STEP = const._STEP
# Top of the troposphere
_ZREF = const._ZMAX


class Zenith:
    """Special value indicating a look vector of "zenith"."""
    pass


def _too_high(positions, zref):
    """Find index of first position higher than zref.

    This is useful when we're trying to cut off integration at the top
    of the troposphere. I calculate the list of all points, then use
    this function to compute the first index above the troposphere, then
    I can cut the list down to just the important points.
    """
    positions_ecef = np.moveaxis(positions, -1, 0)
    positions_lla = np.stack(util.ecef2lla(*positions_ecef))
    high_indices = np.where(positions_lla[2] > zref)[0]
    first_high_index = high_indices[0] if high_indices.size else len(positions)
    return first_high_index


def _get_lengths(look_vecs):
    '''
    Returns the lengths of a vector or set of vectors
    '''
    lengths = np.linalg.norm(look_vecs, axis=-1)
    lengths[~np.isfinite(lengths)] = 0
    return lengths


def _get_steps(lengths):
    '''
    Get the number of integration steps for each path
    '''
    steps = np.array(np.ceil(lengths / _STEP), dtype=np.int64)
    steps[steps < 0] = 0
    return steps


def _getZenithLookVecs(lats, lons, heights, zref = _ZREF):
    '''
    Returns look vectors when Zenith is used
    '''
    return (np.array((util.cosd(lats)*util.cosd(lons),
                              util.cosd(lats)*util.sind(lons),
                              util.sind(lats))).T
                    * (zref - heights)[..., np.newaxis])


def _common_delay(weatherObj, lats, lons, heights, 
                  look_vecs, raytrace, 
                  stepSize = _STEP,
                  verbose = False):
    """
    This function calculates the line-of-sight vectors, estimates the point-wise refractivity
    index for each one, and then integrates to get the total delay in meters. The point-wise
    delay is calculated by interpolating the weatherObj, which contains a weather model with
    wet and hydrostatic refractivity at each weather model grid node, to the points along 
    the ray. The refractivity is integrated along the ray to get the final delay. 
    """
    import interpolator as intprn
    import time

    # Deal with Zenith special value, and non-raytracing method
    if raytrace:
        correction = None
    else:
        correction = 1/util.cosd(look_vecs)
        look_vecs = Zenith

    if look_vecs is Zenith:
        look_vecs = _getZenithLookVecs(lats, lons, heights, zref = _ZREF)

    
    if verbose:
        print('_common_delay: Starting look vector calculation')
        st = time.time()

    mask = np.isnan(heights)
    # Get the integration points along the look vectors
    # First get the length of each look vector, get integration steps along 
    # each, then get the unit vector pointing in the same direction
    lengths = _get_lengths(look_vecs)
    lengths[mask] = np.nan
#    steps = _get_steps(lengths)
    start_positions = np.array(util.lla2ecef(lats, lons, heights)).T
    scaled_look_vecs = look_vecs / lengths[..., np.newaxis]

    if verbose:
        print('_common_delay: The size of look_vecs is {}'.format(np.shape(look_vecs)))
        print('_common_delay: The integration stepsize is {} m'.format(stepSize))

    positions_l, t_points_l,steps = [], [], []
    for i, L in enumerate(lengths):
        # Have to handle the case where there are invalid data
        try:
            thisspace = np.arange(0, L, stepSize)
        except ValueError:
            thisspace = np.array([])

        t_points_l.append(thisspace)
        steps.append(len(thisspace))
        positions_l.append(start_positions[i]
                    + thisspace[..., np.newaxis]*scaled_look_vecs[i])
    if verbose:
        print('_common_delay: Finished steps')
        print('_common_delay: Re-projecting points into native weather model grid')
        

    ecef = pyproj.Proj(proj='geocent')
    newPts = []
    for pnt in positions_l:
        newPts.append(np.stack(pyproj.transform(ecef, 
                                                weatherObj.getProjection(), 
                                                pnt[:,0], pnt[:,1], pnt[:,2]), 
                               axis = -1)
                     )

    if verbose:
        print('_common_delay: Finished re-projecting')
        ft = time.time()
        print('Look vector calculation took {:4.2f} secs'.format(ft-st))
        print('_common_delay: Starting interpolation')
        st = time.time()

    # Define the interpolator
    intFcn= intprn.Interpolator()
    intFcn.setPoints(*weatherObj.getPoints())
    intFcn.setProjection(weatherObj.getProjection())
    intFcn.getInterpFcns(weatherObj.getWetRefractivity(), 
                         weatherObj.getHydroRefractivity(), interpType = 'rgi')

    # call the interpolator on each ray
    wet_pw, hydro_pw = [], []
    for pnt in newPts:
        w, h = intFcn(pnt)
        wet_pw.append(w)
        hydro_pw.append(h)

    if verbose:
        print('_common_delay: Finished interpolation')
        ft = time.time()
        print('Interpolation took {:4.2f} secs'.format(ft-st))
        print('_common_delay: finished point-wise delay calculations')
        print('_common_delay: starting integration')
        st = time.time()

    delays = [] 
    for d in (wet_pw, hydro_pw):
        delays.append(_integrate_delays(stepSize, d))

    if verbose:
        print('_common_delay: Finished integration')
        ft = time.time()
        print('Integration took {:4.2f} secs'.format(ft-st))

    # Finally apply cosine correction if applicable
    if correction is not None:
        delays = [d*correction for d in delays]

    return delays


def _integrate_delays(stepSize, refr):
    '''
    This function gets the actual delays by integrating the refractivity in 
    each node. Refractivity is given in the 'refr' variable. 
    '''

    # integrate the delays to get overall delay
    def int_fcn(y, dx):
        return 1e-6*dx*np.sum(y)

    delays = []
    for ray in refr:
        delays.append(int_fcn(ray, stepSize))
    return delays


def wet_delay(weather, lats, lons, heights, look_vecs, raytrace=True, verbose = False):
    """Compute wet delay along the look vector."""

    if verbose:
        print('wet_delay: Running _common_delay for weather.wet_delay')

    return _common_delay(weather.wet_delay, lats, lons, heights, look_vecs,
                         raytrace, verbose)


def hydrostatic_delay(weather, lats, lons, heights, look_vecs, raytrace=True, verbose = False):
    """Compute hydrostatic delay along the look vector."""

    if verbose:
        print('hydrostatic_delay: Running _common_delay for weather.hydrostatic_delay')

    return _common_delay(weather.hydrostatic_delay, lats, lons, heights,
                         look_vecs, raytrace, verbose)


def delay_over_area(weather, 
                    lat_min, lat_max, lat_res, 
                    lon_min, lon_max, lon_res, 
                    ht_min, ht_max, ht_res, 
                    los=Zenith, 
                    parallel = True, verbose = False):
    """Calculate (in parallel) the delays over an area."""
    lats = np.arange(lat_min, lat_max, lat_res)
    lons = np.arange(lon_min, lon_max, lon_res)
    hts = np.arange(ht_min, ht_max, ht_res)

    if verbose:
        print('delay_over_area: Size of lats: {}'.format(np.shape(lats)))
        print('delay_over_area: Size of lons: {}'.format(np.shape(lons)))
        print('delay_over_area: Size of hts: {}'.format(np.shape(hts)))

    # It's the cartesian product (thanks StackOverflow)
    llas = np.array(np.meshgrid(lats, lons, hts)).T.reshape(-1, 3)
    if verbose:
        print('delay_over_area: Size of llas: {}'.format(np.shape(llas)))

    if verbose:
        print('delay_over_area: running delay_from_grid')

    return delay_from_grid(weather, llas, los, parallel=parallel, verbose = verbose)


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


def delay_from_grid(weather, llas, los, parallel=False, raytrace=True, verbose = False):
    """Calculate delay on every point in a list.

    weather is the weather object, llas is a list of lat, lon, ht points
    at which to calculate delay, and los an array of line-of-sight
    vectors at each point. Pass parallel=True if you want to have real
    speed. If raytrace=True, we'll do raytracing, if raytrace=False,
    we'll do projection.
    """

    # Save the shape so we can restore later, but flatten to make it
    # easier to think about
    real_shape = llas.shape[:-1]
    llas = llas.reshape(-1, 3)
    # los can either be a bunch of vectors or a bunch of scalars. If
    # raytrace, then it's vectors, otherwise scalars. (Or it's Zenith)
    if verbose: 
        if los is Zenith:
           print("LOS is Zenith")
        else:
           print('LOS is not Zenith')

    if los is not Zenith:
        if raytrace:
            los = los.reshape(-1, 3)
        else:
            los = los.flatten()

    lats, lons, hts = np.moveaxis(llas, -1, 0)

    # Call _common_delay to compute both the hydrostatic and wet delays
    wet, hydro = _common_delay(weather, lats, lons, hts, los, raytrace = raytrace, verbose = verbose)
#    hydro = hydrostatic_delay(weather, lats, lons, hts, los,
#    wet = wet_delay(weather, lats, lons, hts, los, raytrace=raytrace, verbose = verbose)

    # Restore shape
    hydro, wet = np.stack((hydro, wet)).reshape((2,) + real_shape)

    return hydro, wet


def delay_from_files(weather, lat, lon, ht, parallel=False, los=Zenith,
                     raytrace=True, verbose = False):
    """Read location information from files and calculate delay."""
    lats = util.gdal_open(lat)
    lons = util.gdal_open(lon)
    hts = util.gdal_open(ht)

    if los is not Zenith:
        incidence, heading = util.gdal_open(los)
        if raytrace:
            los = losreader.los_to_lv(
                incidence, heading, lats, lons, hts, _ZREF).reshape(-1, 3)
        else:
            los = incidence.flatten()

    # We need the three to be the same shape so that we know what to
    # reshape hydro and wet to. Plus, them being different sizes
    # indicates a definite user error.
    if not lats.shape == lons.shape == hts.shape:
        raise ValueError(f'lat, lon, and ht should have the same shape, but '
                         'instead lat had shape {lats.shape}, lon had shape '
                         '{lons.shape}, and ht had shape {hts.shape}')

    llas = np.stack((lats.flatten(), lons.flatten(), hts.flatten()), axis=1)
    hydro, wet = delay_from_grid(weather, llas, los,
                                 parallel=parallel, raytrace=raytrace, verbose = verbose)
    hydro, wet = np.stack((hydro, wet)).reshape((2,) + lats.shape)
    return hydro, wet


def _tropo_delay_with_values(los, lats, lons, hts, 
                             weather, zref, 
                             time, 
                             raytrace = True,
                             parallel = True, verbose = False):
    """Calculate troposphere delay from processed command-line arguments."""
    # LOS
    if los is None:
        los = Zenith
    else:
        los = losreader.infer_los(los, lats, lons, hts, zref)

    # We want to test if any shapes are different
    if (not hts.shape == lats.shape == lons.shape
            or los is not Zenith and los.shape[:-1] != hts.shape):
        raise ValueError(
         'I need lats, lons, heights, and los to all be the same shape. ' +
         'lats had shape {}, lons had shape {}, '.format(lats.shape, lons.shape)+
         'heights had shape {}, and los had shape {}'.format(hts.shape,los.shape))

    if verbose: 
        print('_tropo_delay_with_values: called delay_from_grid')

    # Do the calculation
    llas = np.stack((lats, lons, hts), axis=-1)
    hydro, wet = delay_from_grid(weather, llas, los, parallel=parallel,
                                 raytrace=raytrace, verbose = verbose)
    return hydro, wet


def get_weather_and_nodes(model, filename, zmin=None):
    """Look up weather information from a model and file.

    We use the module.load method to load the weather model file, but
    we'll also create a weather model object for it.
    """
    xs, ys, proj, t, q, z, lnsp = model.load(filename)
    return (reader.read_model_level(module, xs, ys, proj, t, q, z, lnsp, zmin),
            xs, ys, proj)


def tropo_delay(los = None, lat = None, lon = None, 
                heights = None, 
                weather = None, 
                zref = 15000, 
                out = None, 
                time = None,
                outformat='ENVI', 
                parallel=True,
                verbose = False):
    """Calculate troposphere delay from command-line arguments.

    We do a little bit of preprocessing, then call
    _tropo_delay_with_values. Then we'll write the output to the output
    file.
    """
    import pyproj

    if out is None:
        out = os.getcwd()

    # Make weather
    weather_type = weather['type']
    weather_files = weather['files']
    weather_fmt = weather['name']

    # For later
    str1 = time.isoformat() + "_" if time is not None else ""
    str2 = "z" if los is None else "s" 
    str3 = 'td.{}'.format(outformat)
    hydroname, wetname = (
        '{}_{}_'.format(weather_fmt, dtyp) + str1 + str2 + str3
        for dtyp in ('hydro', 'wet'))

    hydro_file_name = os.path.join(out, hydroname)
    wet_file_name = os.path.join(out, wetname)

    # set_geo_info should be a list of functions to call on the dataset,
    # and each will do some bit of work
    set_geo_info = list()

    # Lat, lon
    if lat is None:
        # They'll get set later with weather
        lats = lons = None
        latproj = lonproj = None
    else:
        lats, latproj = util.gdal_open(lat, returnProj = True)
        lons, lonproj = util.gdal_open(lon, returnProj = True)

    # set_geo_info should be a list of functions to call on the dataset,
    # and each will do some bit of work
    set_geo_info = list()
    if lat is not None:
        def geo_info(ds):
            ds.SetMetadata({'X_DATASET': os.path.abspath(lat), 'X_BAND': '1',
                            'Y_DATASET': os.path.abspath(lon), 'Y_BAND': '1'})
        set_geo_info.append(geo_info)
    # Is it ever possible that lats and lons will actually have embedded
    # projections?
    if latproj:
        def geo_info(ds):
            ds.SetProjection(latproj)
        set_geo_info.append(geo_info)
    elif lonproj:
        def geo_info(ds):
            ds.SetProjection(lonproj)
        set_geo_info.append(geo_info)

    height_type, height_info = heights
    if verbose:
        print('Type of height: {}'.format(height_type))
        print('Type of weather model: \n {}'.format(weather_type))
        if weather_files is not None:
            print('{} weather files'.format(len(weather_files)))
        print('Weather format: {}'.format(weather_fmt))

    if weather_type == 'wrf':
        import wrf
        weather = wrf.load(*weather_files)

        # Let lats and lons to weather model nodes if necessary
        if lats is None:
            lats, lons = wrf.wm_nodes(*weather_files)
    elif weather_type == 'pickle':
        weather = util.pickle_load('weatherObj.dat')
    else:
        weather_model = weather_type
        if weather_files is None:
            if lats is None:
                raise ValueError(
                    'Unable to infer lats and lons if you also want me to '
                    'download the weather model')
            if verbose:
                f = os.path.join(out, 'weather_model.dat')
                weather_model.fetch(lats, lons, time, f)
                weather_model.load(f)
                weather = weather_model # Need to maintain backwards compatibility at the moment
                print(weather)
                try:
                    util.pickle_dump(weather, 'weatherObj.dat')
                except:
                    pass
                weather.plot()
            else:
                with tempfile.NamedTemporaryFile() as f:
                    weather_model.fetch(lats, lons, time, f)
                    weather_model.load(f)
                    weather = weather_model
        else:
            weather, xs, ys, proj = weather_model.weather_and_nodes(
                weather_files)
            if lats is None:
                def geo_info(ds):
                    ds.SetProjection(str(proj))
                    ds.SetGeoTransform((xs[0], xs[1] - xs[0], 0, ys[0], 0,
                                        ys[1] - ys[0]))
                set_geo_info.append(geo_info)
                lla = pyproj.Proj(proj='latlong')
                xgrid, ygrid = np.meshgrid(xs, ys, indexing='ij')
                lons, lats = pyproj.transform(proj, lla, xgrid, ygrid)

    # Height
    if height_type == 'dem':
        hts = util.gdal_open(height_info)
    elif height_type == 'lvs':
        hts = height_info
    elif height_type == 'download':
        hts = demdownload.download_dem(lats, lons)
    else:
        raise ValueError(f'Unexpected height_type {repr(height_type)}')

    # Pretty different calculation depending on whether they specified a
    # list of heights or just a DEM
    if height_type == 'lvs':
        shape = (len(hts),) + lats.shape
        total_hydro = np.zeros(shape)
        total_wet = np.zeros(shape)
        for i, ht in enumerate(hts):
            hydro, wet = _tropo_delay_with_values(
                los, lats, lons, np.broadcast_to(ht, lats.shape), weather,
                zref, time, parallel=parallel, verbose = verbose)
            total_hydro[i] = hydro
            total_wet[i] = wet

        if outformat == 'hdf5':
            raise NotImplemented
        else:
            drv = gdal.GetDriverByName(outformat)
            hydro_ds = drv.Create(
                hydro_file_name, total_hydro.shape[2],
                total_hydro.shape[1], len(hts), gdal.GDT_Float64)
            for lvl, (hydro, ht) in enumerate(zip(total_hydro, hts), start=1):
                band = hydro_ds.GetRasterBand(lvl)
                band.SetDescription(str(ht))
                band.WriteArray(hydro)
            for f in set_geo_info:
                f(hydro_ds)
            hydro_ds = None

            wet_ds = drv.Create(
                wet_file_name, total_wet.shape[2],
                total_wet.shape[1], len(hts), gdal.GDT_Float64)
            for lvl, (wet, ht) in enumerate(zip(total_wet, hts), start=1):
                band = wet_ds.GetRasterBand(lvl)
                band.SetDescription(str(ht))
                band.WriteArray(wet)
            for f in set_geo_info:
                f(wet_ds)
            wet_ds = None

    else:
        hydro, wet = _tropo_delay_with_values(
            los, lats, lons, hts, weather, zref, time, parallel = parallel, verbose = verbose)

        # Write the output file
        # TODO: maybe support other files than ENVI
        if outformat == 'hdf5':
            raise NotImplemented
        else:
            drv = gdal.GetDriverByName(outformat)
            hydro_ds = drv.Create(
                hydro_file_name, hydro.shape[1], hydro.shape[0],
                1, gdal.GDT_Float64)
            hydro_ds.GetRasterBand(1).WriteArray(hydro)
            for f in set_geo_info:
                f(hydro_ds)
            hydro_ds = None
            wet_ds = drv.Create(
                wet_file_name, wet.shape[1], wet.shape[0], 1,
                gdal.GDT_Float64)
            wet_ds.GetRasterBand(1).WriteArray(wet)
            for f in set_geo_info:
                f(wet_ds)
            wet_ds = None

    return hydro_file_name, wet_file_name
