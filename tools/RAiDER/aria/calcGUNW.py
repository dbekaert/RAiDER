"""
Calculate the interferometric phase from the 4 delays files of a GUNW
Write it to disk
"""
import os
import xarray as xr
import numpy as np
from RAiDER.utilFcns import rio_open
from RAiDER.logger import logger
from datetime import datetime
import h5py
from netCDF4 import Dataset

## ToDo:
    # Check difference direction

TROPO_GROUP = 'science/grids/corrections/external/troposphere'
TROPO_NAMES = ['troposphereWet', 'troposphereHydrostatic']


def compute_delays(cube_filenames:list, wavelength):
    """ Difference the delays and convert to radians. Return xr dataset. """
    # parse date from filename
    dct_delays = {}
    for f in cube_filenames:
        date = datetime.strptime(os.path.basename(f).split('_')[2], '%Y%m%dT%H%M%S')
        dct_delays[date] = f

    sec, ref = sorted(dct_delays.keys())

    wet_delays = []
    hyd_delays = []
    for dt in [ref, sec]:
        path = dct_delays[dt]
        with xr.open_dataset(path) as ds:
            da_wet   = ds['wet']
            da_hydro = ds['hydro']

            wet_delays.append(da_wet)
            hyd_delays.append(da_hydro)

    scale    = float(wavelength) / (4 * np.pi)
    wetDelay = (wet_delays[0] - wet_delays[1]) * scale
    hydDelay = (hyd_delays[0] - hyd_delays[1]) * scale

    ds_ifg   = xr.open_dataset(path).copy()
    del ds_ifg['wet'], ds_ifg['hydro']

    ds_ifg[TROPO_NAMES[0]] = wetDelay
    ds_ifg[TROPO_NAMES[1]] = hydDelay

    model = os.path.basename(path).split('_')[0]
    ref   = f"{ref.date().strftime('%Y%m%d')}"
    sec   = f"{sec.date().strftime('%Y%m%d')}"

    attrs = {'model': model,
             'history': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             'method': 'ray tracing', 'units': 'radians',
             }

    names = {TROPO_NAMES[0]: 'tropoWet', TROPO_NAMES[1]: 'tropoHyd'}
    for k, v in names.items():
        ds_ifg[k] = ds_ifg[k].assign_attrs(attrs)
        ds_ifg[k] = ds_ifg[k].assign_attrs(long_name=k, short_name=v)

    return ds_ifg


def tropo_gunw_inf(cube_filenames:dict, path_gunw:str, wavelength, out_dir:str, update_flag:bool):
    """ Calculate interferometric phase delay

    Requires:
        list with filename of delay cube for ref and sec date (netcdf)
        path to the gunw file
        wavelength (units: m)
        output directory (where to store the delays)
        update_flag (to write into the GUNW or not)
    """
    ds_ifg = compute_delays(cube_filenames, wavelength)
    da     = ds_ifg[TROPO_NAMES[0]] # for metadata

    # write the interferometric delay to disk
    ref, sec = os.path.basename(path_gunw).split('-')[6].split('_')
    model    = da.model
    dst      = os.path.join(out_dir, f'{model}_interferometric_{ref}_{sec}.nc')
    ds_ifg.to_netcdf(dst)
    logger.info ('Wrote interferometric delays to: %s', dst)

    ## optionally update netcdf with the interferometric delay
    if update_flag:
        ## first need to delete the variable; only can seem to with h5
        with h5py.File(path_gunw, 'a') as h5:
            for k in TROPO_GROUP.split():
                h5 = h5[k]
            del h5[TROPO_NAMES[0]]
            del h5[TROPO_NAMES[1]]

        with Dataset(path_gunw, mode='a') as ds:
            ds_grp = ds[TROPO_GROUP]

            for dim in 'z y x'.split():
                ## dimension may already exist if updating
                try:
                    ds_grp.createDimension(dim, len(ds_ifg.coords[dim]))
                except:
                    pass

            for name in TROPO_NAMES:
                v  = ds_grp.createVariable(name, float, 'z y x'.split())
                da = ds_ifg[name]
                v[:] = da.data
                v.setncatts(da.attrs)

        logger.info('Updated %s group in: %s', os.path.basename(TROPO_GROUP), path_gunw)


if __name__ == '__main__':
    main()
