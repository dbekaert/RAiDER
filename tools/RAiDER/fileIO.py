#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def writeWeatherLevelsToFile(totalhydro, totalwet, outFormat, hydro_file_name, wet_file_name):
    '''
    Need to implement a separate file writing 
    routine for list-like outputs
    '''
    if outformat == 'hdf5':
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

       # writing the heights
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

    else:
        import util
        util.writeArrayToRaster(totalhydro, hydro_file_name, noDataValue = 0.)
        util.writeArrayToRaster(totalwet, wet_file_name, noDataValue = 0.)


def writeStationDelaysToFile(totalhydro, totalwet, hydro_file_name, wet_file_name, outFormat):
    '''
    Write station delays to a file. 
    '''
    if outFormat == 'netcdf':
       # do something 
       pass
    elif outFormat=='hdf5':
       # do something else
       pass
    else:
       raise RuntimeError('I cannot write to the format {}'.format(outFormat))

