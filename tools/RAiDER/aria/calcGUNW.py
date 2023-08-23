"""
Calculate the interferometric phase from the 4 delays files of a GUNW
Write it to disk
"""
import os
import xarray as xr
import numpy as np
import RAiDER
from RAiDER.utilFcns import rio_open
from RAiDER.logger import logger
from datetime import datetime
import h5py
from netCDF4 import Dataset

## ToDo:
    # Check difference direction

TROPO_GROUP = 'science/grids/corrections/external/troposphere'
TROPO_NAMES = ['troposphereWet', 'troposphereHydrostatic']
DIM_NAMES   = ['heightsMeta', 'latitudeMeta', 'longitudeMeta']


def compute_delays_slc(cube_filenames:list, wavelength):
    """ Get delays and convert to radians. Return xr dataset. """
    # parse date from filename
    dct_delays = {}
    for f in cube_filenames:
        date = datetime.strptime(os.path.basename(f).split('_')[2], '%Y%m%dT%H%M%S')
        dct_delays[date] = f

    sec, ref = sorted(dct_delays.keys())

    wet_delays  = []
    hyd_delays  = []
    phase2range = (-4 * np.pi) / float(wavelength)
    for dt in [ref, sec]:
        path = dct_delays[dt]
        with xr.open_dataset(path) as ds:
            da_wet   = ds['wet'] * phase2range
            da_hydro = ds['hydro'] * phase2range

            wet_delays.append(da_wet)
            hyd_delays.append(da_hydro)

            crs = da_wet.rio.crs
            gt  = da_wet.rio.transform()

    chunk_sizes = da_wet.shape[0], da_wet.shape[1]/3, da_wet.shape[2]/3

    # open one to copy and store new data
    ds_slc   = xr.open_dataset(path).copy()
    encoding = ds_slc['wet'].encoding # chunksizes and fill value
    encoding['contiguous'] = False
    encoding['_FillValue'] = 0.
    encoding['chunksizes'] = tuple([np.floor(cs) for cs in chunk_sizes])
    del ds_slc['wet'], ds_slc['hydro']

    for i, key in enumerate('reference secondary'.split()):
        ds_slc[f'{key}_{TROPO_NAMES[0]}'] = wet_delays[i]
        ds_slc[f'{key}_{TROPO_NAMES[1]}'] = hyd_delays[i]

    model = os.path.basename(path).split('_')[0]

    attrs = {
             'units': 'radians',
             'grid_mapping': 'crs',
             }

    ## no data (fill value?) chunk size?
    for name in TROPO_NAMES:
        for key in 'reference secondary'.split():
            descrip  = f"Delay due to {name.lstrip('troposphere')} component of troposphere"
            da_attrs = {**attrs,  'description':descrip,
                        'long_name':name, 'standard_name':name,
                        'RAiDER version': RAiDER.__version__,
                        }
            ds_slc[f'{key}_{name}'] = ds_slc[f'{key}_{name}'].assign_attrs(da_attrs)
            ds_slc[f'{key}_{name}'].encoding = encoding

    ds_slc = ds_slc.assign_attrs(model=model, method='ray tracing')

    ## force these to float32 to prevent stitching errors
    coords = {coord:ds_slc[coord].astype(np.float32) for coord in ds_slc.coords}
    ds_slc = ds_slc.assign_coords(coords)

    return ds_slc.rename(z=DIM_NAMES[0], y=DIM_NAMES[1], x=DIM_NAMES[2])


def update_gunw_slc(path_gunw:str, ds_slc):
    """ Update the path_gunw file using the slc delays in ds_slc """
    ## first need to delete the variable; only can seem to with h5
    with h5py.File(path_gunw, 'a') as h5:
        for k in TROPO_GROUP.split():
            h5 = h5[k]
        # in case GUNW has already been updated once before
        try:
            del h5[TROPO_NAMES[0]]
            del h5[TROPO_NAMES[1]]
        except KeyError:
            pass

        for k in 'crs'.split():
            if k in h5.keys():
                del h5[k]


    with Dataset(path_gunw, mode='a') as ds:
        ds_grp    = ds[TROPO_GROUP]
        ds_grp.createGroup(ds_slc.attrs['model'].upper())
        ds_grp_wm = ds_grp[ds_slc.attrs['model'].upper()]


        ## create and store new data e.g., corrections/troposphere/GMAO/reference/troposphereWet
        for rs in 'reference secondary'.split():
            ds_grp_wm.createGroup(rs)
            ds_grp_rs = ds_grp_wm[rs]

            ## create the new dimensions e.g., corrections/troposphere/GMAO/reference/latitudeMeta
            for dim in DIM_NAMES:
                ## dimension may already exist if updating
                try:
                    ds_grp_rs.createDimension(dim, len(ds_slc.coords[dim]))
                    ## necessary for transform
                    v  = ds_grp_rs.createVariable(dim, np.float32, dim)
                    v[:] = ds_slc[dim]
                    v.setncatts(ds_slc[dim].attrs)

                except RuntimeError:
                    pass

            ## add the projection if it doesnt exist
            try:
                v_proj = ds_grp_rs.createVariable('crs', 'i')
            except RuntimeError:
                v_proj = ds_grp_rs['crs']
            v_proj.setncatts(ds_slc["crs"].attrs)

            ## update the actual tropo data
            for name in TROPO_NAMES:
                da        = ds_slc[f'{rs}_{name}']
                nodata    = da.encoding['_FillValue']
                chunksize = da.encoding['chunksizes']

                ## in case updating
                try:
                    v    = ds_grp_rs.createVariable(name, np.float32, DIM_NAMES,
                                        chunksizes=chunksize, fill_value=nodata)
                except RuntimeError:
                    v    = ds_grp_rs[name]

                v[:] = da.data
                v.setncatts(da.attrs)


    logger.info('Updated %s group in: %s', os.path.basename(TROPO_GROUP), path_gunw)
    return


def update_gunw_version(path_gunw):
    """ temporary hack for updating version to test aria-tools """
    with Dataset(path_gunw, mode='a') as ds:
        ds.version = '1c'
    return


### ------------------------------------------------------------- main function
def tropo_gunw_slc(cube_filenames:list, path_gunw:str, wavelength, out_dir:str, update_gunw:bool):
    """ Calculate ref/sec phase delay

    Requires:
        list with filename of delay cube for ref and sec date (netcdf)
        path to the gunw file
        wavelength (units: m)
        output directory (where to store the delays)
    """
    os.makedirs(out_dir, exist_ok=True)

    ds_slc = compute_delays_slc(cube_filenames, wavelength)
    da     = ds_slc[f'reference_{TROPO_NAMES[0]}'] # for metadata
    model  = ds_slc.model

    # write the interferometric delay to disk
    ref, sec = os.path.basename(path_gunw).split('-')[6].split('_')
    mid_time = os.path.basename(path_gunw).split('-')[7]
    dst      = os.path.join(out_dir, f'{model}_interferometric_{ref}-{sec}_{mid_time}.nc')
    ds_slc.to_netcdf(dst)
    logger.info ('Wrote slc delays to: %s', dst)

    ## optionally update netcdf with the slc delay
    update_gunw_slc(path_gunw, ds_slc)

    ## temp
    update_gunw_version(path_gunw)


if __name__ == '__main__':
    main()
