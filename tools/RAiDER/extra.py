#
#        else:
#            weather, xs, ys, proj = weather_model.weather_and_nodes(
#                weather_files)
#            if lats is None:
#                def geo_info(ds):
#                    ds.SetProjection(str(proj))
#                    ds.SetGeoTransform((xs[0], xs[1] - xs[0], 0, ys[0], 0,
#                                        ys[1] - ys[0]))
#                set_geo_info.append(geo_info)
#                lla = pyproj.Proj(proj='latlong')
#                xgrid, ygrid = np.meshgrid(xs, ys, indexing='ij')
#                lons, lats = pyproj.transform(proj, lla, xgrid, ygrid)
#
#
#
#
#        # use the weather model nodes in case the grid is not specififed
#        if lats is None:
#            # weather model projection
#            proj = weather_model.getProjection()
#            # weather model grid (3D cube= x,y,z)
#            xs = weather_model._xs 
#            ys = weather_model._ys
#            # only using the grid from one dimension as z is a replicate
#            xs = xs[:,:,0]
#            ys = ys[:,:,0]
#
#            # for geo transform make sure that we take right slice through the x and y grids
#            if len(np.unique(xs))==len(xs[:,0]):
#                xs_vector = xs[:,0]
#                ys_vector = np.transpose(ys[0,:])
#            elif len(np.unique(xs))==len(xs[0,:]):
#                ys_vector = ys[:,0]
#                xs_vector = np.transpose(xs[0,:])
#
#            # code up the default projectoon to be WGS84 for local processing/ delay calculation
#            lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
#            lons, lats = pyproj.transform(proj, lla,xs,ys)
#    
#            def geo_info(ds):
#                ds.SetProjection(str(proj))
#                ds.SetGeoTransform((xs_vector[0], xs_vector[1] - xs_vector[0], 0, ys_vector[0], 0, ys_vector[1] - ys_vector[0]))
#                set_geo_info.append(geo_info)
#    
#
#
#
#
#def _helper(tup):
#    
#    return _compute_ray(tup[0], tup[1], tup[2], tup[3])
#    #return _compute_ray(L, S, V, stepSize)
#
#def _get_rays_p(lengths, stepSize, start_positions, scaled_look_vecs, Nproc = 4):
#    import multiprocessing as mp
#
#    # setup for multiprocessing
#    data = zip(lengths, start_positions, scaled_look_vecs, [stepSize]*len(lengths))
#
#    pool = mp.Pool(Nproc)
#    positions_l = pool.map(helper, data)
#    return positions_l
#
#def _get_rays_d(lengths, stepSize, start_positions, scaled_look_vecs, Nproc = 2):
#   import dask.bag as db
#   L = db.from_sequence(lengths)
#   S = db.from_sequence(start_positions)
#   Sv = db.from_sequence(scaled_look_vecs)
#   Ss = db.from_sequence([stepSize]*len(lengths))
#
#   # setup for multiprocessing
#   data = db.zip(L, S, Sv, Ss)
#
#   positions_l = db.map(helper, data)
#   return positions_l.compute()
#
#
#
#TODO: the following three fcns are unclear if/how they are needed. 
# likely need to see how they work with tropo_delay
#def delay_over_area(weather, 
#                    lat_min, lat_max, lat_res, 
#                    lon_min, lon_max, lon_res, 
#                    ht_min, ht_max, ht_res, 
#                    los=Zenith, 
#                    parallel = True, verbose = False):
#    """Calculate (in parallel) the delays over an area."""
#    lats = np.arange(lat_min, lat_max, lat_res)
#    lons = np.arange(lon_min, lon_max, lon_res)
#    hts = np.arange(ht_min, ht_max, ht_res)
#
#    if verbose:
#        print('delay_over_area: Size of lats: {}'.format(np.shape(lats)))
#        print('delay_over_area: Size of lons: {}'.format(np.shape(lons)))
#        print('delay_over_area: Size of hts: {}'.format(np.shape(hts)))
#
#    # It's the cartesian product (thanks StackOverflow)
#    llas = np.array(np.meshgrid(lats, lons, hts)).T.reshape(-1, 3)
#    if verbose:
#        print('delay_over_area: Size of llas: {}'.format(np.shape(llas)))
#
#    if verbose:
#        print('delay_over_area: running delay_from_grid')
#
#    return delay_from_grid(weather, llas, los, parallel=parallel, verbose = verbose)
#
#
#def delay_from_files(weather, lat, lon, ht, zref = None, parallel=False, los=Zenith,
#                     raytrace=True, verbose = False):
#    """
#    Read location information from files and calculate delay.
#    """
#    if zref is None:
#       zref = _ZREF
#
#    lats = utilFcns.gdal_open(lat)
#    lons = utilFcns.gdal_open(lon)
#    hts = utilFcns.gdal_open(ht)
#
#    if los is not Zenith:
#        incidence, heading = utilFcns.gdal_open(los)
#        if raytrace:
#            los = losreader.los_to_lv(
#                incidence, heading, lats, lons, hts, zref).reshape(-1, 3)
#        else:
#            los = incidence.flatten()
#
#    # We need the three to be the same shape so that we know what to
#    # reshape hydro and wet to. Plus, them being different sizes
#    # indicates a definite user error.
#    if not lats.shape == lons.shape == hts.shape:
#        raise ValueError('lat, lon, and ht should have the same shape, but ' + 
#                         'instead lat had shape {}, lon had shape '.format(lats.shape) + 
#                         '{}, and ht had shape {}'.format(lons.shape,hts.shape))
#
#    llas = np.stack((lats.flatten(), lons.flatten(), hts.flatten()), axis=1)
#    hydro, wet = delay_from_grid(weather, llas, los,
#                                 parallel=parallel, raytrace=raytrace, verbose = verbose)
#    hydro, wet = np.stack((hydro, wet)).reshape((2,) + lats.shape)
#    return hydro, wet
#
#
#def get_weather_and_nodes(model, filename, zmin=None):
#    """Look up weather information from a model and file.
#
#    We use the module.load method to load the weather model file, but
#    we'll also create a weather model object for it.
#    """
#    # TODO: Need to check how this fcn will fit into the new framework
#    xs, ys, proj, t, q, z, lnsp = model.load(filename)
#    return (reader.read_model_level(module, xs, ys, proj, t, q, z, lnsp, zmin),
#            xs, ys, proj)
#
#

# Below was a part of weatherModel.py and used to restrict the model in Z
# I took it out because there are not that many levels above and seems easier to 
# keep them all then try to put in a cutoff


        # Now remove any model level fully above zmax
#        max_level_needed = utilFcns.getMaxModelLevel(self._zs, self._zmax, 'g') 
#        levInd = range(0,max_level_needed + 1)
        

#        if self._humidityType == 'q':
#            self._q = self._q[...,levInd]
#        else:
#            self._rh = self._rh[...,levInd]
#
#        self._zs = self._zs[...,levInd]
#        self._xs = self._xs[...,levInd]
#        self._ys = self._ys[...,levInd]
#        self._p = self._p[...,levInd]
#        self._t = self._t[...,levInd]
#        self._e = self._e[...,levInd]
#        self._wet_refractivity = self._wet_refractivity[...,levInd]
#        self._hydrostatic_refractivity=self._hydrostatic_refractivity[...,levInd]


