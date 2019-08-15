
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




        # use the weather model nodes in case the grid is not specififed
        if lats is None:
            # weather model projection
            proj = weather_model.getProjection()
            # weather model grid (3D cube= x,y,z)
            xs = weather_model._xs 
            ys = weather_model._ys
            # only using the grid from one dimension as z is a replicate
            xs = xs[:,:,0]
            ys = ys[:,:,0]

            # for geo transform make sure that we take right slice through the x and y grids
            if len(np.unique(xs))==len(xs[:,0]):
                xs_vector = xs[:,0]
                ys_vector = np.transpose(ys[0,:])
            elif len(np.unique(xs))==len(xs[0,:]):
                ys_vector = ys[:,0]
                xs_vector = np.transpose(xs[0,:])

            # code up the default projectoon to be WGS84 for local processing/ delay calculation
            lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
            lons, lats = pyproj.transform(proj, lla,xs,ys)
    
            def geo_info(ds):
                ds.SetProjection(str(proj))
                ds.SetGeoTransform((xs_vector[0], xs_vector[1] - xs_vector[0], 0, ys_vector[0], 0, ys_vector[1] - ys_vector[0]))
                set_geo_info.append(geo_info)
    




def _helper(tup):
    
    return _compute_ray(tup[0], tup[1], tup[2], tup[3])
    #return _compute_ray(L, S, V, stepSize)

def _get_rays_p(lengths, stepSize, start_positions, scaled_look_vecs, Nproc = 4):
    import multiprocessing as mp

    # setup for multiprocessing
    data = zip(lengths, start_positions, scaled_look_vecs, [stepSize]*len(lengths))

    pool = mp.Pool(Nproc)
    positions_l = pool.map(helper, data)
    return positions_l

def _get_rays_d(lengths, stepSize, start_positions, scaled_look_vecs, Nproc = 2):
   import dask.bag as db
   L = db.from_sequence(lengths)
   S = db.from_sequence(start_positions)
   Sv = db.from_sequence(scaled_look_vecs)
   Ss = db.from_sequence([stepSize]*len(lengths))

   # setup for multiprocessing
   data = db.zip(L, S, Sv, Ss)

   positions_l = db.map(helper, data)
   return positions_l.compute()


