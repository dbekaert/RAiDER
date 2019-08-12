#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def writeWeatherLevelsToFile():
    '''
    Need to implement a separate file writing 
    routine for list-like outputs
    '''
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



if outformat == 'hdf5' or outformat=='h5' or outformat =='hdf':
    if os.path.isfile(hydro_file_name):
        os.remove(hydro_file_name)

    # for hdf5 we will hard-code only the lon-lat grid for now
    # for hdf and netcdf typically the dimensions are listed as z,y,x
    # coordinates are provided as vectors for regular grids
    
    # hydro delay
    hydro_ds = h5py.File(hydro_file_name,'w')
    # generating hydro dataset
    hdftotal_hydro = np.swapaxes(total_hydro,1,2)
    dset = hydro_ds.create_dataset("hydro",data=hdftotal_hydro,fillvalue=0,compression="gzip")
    dset.attrs["Description"]='Hydro delay in meters'
    # generating lon lat dataset
    if len(np.unique(lons))==len(lons[:,0]):
        hdflons = lons[:,0]
        hdflats = np.transpose(lats[0,:])
        dset = hydro_ds.create_dataset("lons",data=hdflons,compression="gzip")
        dset = hydro_ds.create_dataset("lats",data=hdflats,compression="gzip")
    elif len(np.unique(lons))==len(lons[0,:]):
        hdflons = np.transpose(lons[0,:])
        hdflats = lats[:,0]
        dset = hydro_ds.create_dataset("lons",data=hdflons,compression="gzip")
        dset = hydro_ds.create_dataset("lats",data=hdflats,compression="gzip")
    else:
        dset = hydro_ds.create_dataset("lons_grid",data=lons,fillvalue=0,compression="gzip")
        dset = hydro_ds.create_dataset("lats_grid",data=lats,fillvalue=0,compression="gzip")
    # writing teh heights
    dset = hydro_ds.create_dataset("heights",data=hts,compression="gzip")
    # create the projection string
    proj4= 'EPSG:{0}'.format(int(4326))
    hydro_ds.create_dataset('projection', data=[proj4.encode('utf-8')], dtype='S200')
    # close the file
    hydro_ds.close()
    # hydro delay
    wet_ds = h5py.File(wet_file_name,'w')
    # generating hydro dataset
    hdftotal_wet = np.swapaxes(total_wet,1,2)
    dset = wet_ds.create_dataset("wet",data=hdftotal_wet,fillvalue=0,compression="gzip")
    dset.attrs["Description"]='Wet delay in meters'
    # generating lon lat dataset
    if len(np.unique(lons))==len(lons[:,0]):
        hdflons = lons[:,0]
        hdflats = np.transpose(lats[0,:])
        dset = wet_ds.create_dataset("lons",data=hdflons,compression="gzip")
        dset = wet_ds.create_dataset("lats",data=hdflats,compression="gzip")
    elif len(np.unique(lons))==len(lons[0,:]):
        hdflons = np.transpose(lons[0,:])
        hdflats = lats[:,0]
        dset = wet_ds.create_dataset("lons",data=hdflons,compression="gzip")
        dset = wet_ds.create_dataset("lats",data=hdflats,compression="gzip")
    else:
        dset = wet_ds.create_dataset("lons_grid",data=lons,fillvalue=0,compression="gzip")
        dset = wet_ds.create_dataset("lats_grid",data=lats,fillvalue=0,compression="gzip")
    # writing the heights
    dset = wet_ds.create_dataset("heights",data=hts,compression="gzip")
    # create teh projection string                                                                    
    wet_ds.create_dataset('projection', data=[proj4.encode('utf-8')], dtype='S200')
    # close the file
    wet_ds.close()

