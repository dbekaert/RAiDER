"""
Calculate the interferometric phase from the 4 delays files of a GUNW
Write it to disk
"""
import os
import xarray as xr
import numpy as np
from RAiDER.utilFcns import rio_open
from RAiDER.logger import logger

## ToDo:
    # Write back to GUNW
    # Check difference direction
    # Capture Metadata
        # capture method (proj slant zenith) in metadata
        # capture model
        # capture description of variable
        # capture units # radians
        # short name, long name
        # projection


def tropo_gunw_inf(dct_delays:dict, path_gunw:str, wavelength, out_dir:str, update_flag:bool):
    """ Calculate interferometric phase delay

    Requires:
        dictionary of date: path to cube of delays in netcdf format
        path to the gunw file
        wavelength (units: m)
        output directory (where to store the delays)
        update_flag (to write into the GUNW or not)
    """

    sec, ref = sorted(dct_delays.keys())

    wet_delays = []
    hyd_delays = []
    for dt in [ref, sec]:
        path = dct_delays[dt][0] # both the same for cube (tropo)
        with xr.open_dataset(path) as ds:
            da_wet   = ds['wet']
            da_hydro = ds['hydro']

            wet_delays.append(da_wet)
            hyd_delays.append(da_hydro)

    scale    = wavelength / (4 * np.pi)
    wetDelay = (wet_delays[0] - wet_delays[1]) * scale
    hydDelay = (hyd_delays[0] - hyd_delays[1]) * scale

    ds_ifg  = xr.open_dataset(path).copy()
    ds_ifg['wet']   = wetDelay
    ds_ifg['hydro'] = hydDelay
    model = os.path.basename(path).split('_')[0]
    dates = f"{ref.date().strftime('%Y%m%d')}_{sec.date().strftime('%Y%m%d')}"
    dst   = os.path.join(out_dir, f'{model}_interferometric_{dates}.nc')
    ds.to_netcdf(dst)
    logger.info ('Wrote interferometric delays to: %s', dst)

    ## optionally update netcdf
    if update_flag:
        # names   = 'troposphereWet troposphereHydrostatic'.split()
        # lst_das = []
        # for name, delays in zip(names, [wetDelay, hydDelay, totDelay]):
        #     da = xr.DataArray(delays, name=name, dims='latitude longitude'.split(),
        #             coords={'latitude': lats, 'longitude': lons})
        # ds  = xr.merge(lst_ds)
        pass


if __name__ == '__main__':
    main()
