"""
Calculate the interferometric phase from the 4 delays files of a GUNW
Write it to disk
"""
import os
import xarray as xr
import numpy as np
from RAiDER.utilFcns import rio_open
from RAiDER.logger import logger

# capture method (proj slant zenith) in metadata
# capture model
# capture description of variable
# capture units # radians
# short name, long name
# projection


def main(dct_delays:dict, path_gunw:str, wavelength, out_dir:str, update_flag):
    """ Calculate interferometric phase delay

    Requires:
        4 delay files; wet and hydro for each date
        path to the gunw file
        wavelength
        output directory (where to store the delays)
        update_flag (to write into the GUNW or not)
    """
    sec, ref = sorted(dct_delays.keys())

    wet_delays = []
    hyd_delays = []
    tot_delays = []
    for dt in [ref, sec]:
        lst_paths = dct_delays[dt]
        arr_wet   = rio_open(lst_paths[0])
        arr_hydro = rio_open(lst_paths[1])
        wet_delays.append(arr_wet)
        hyd_delays.append(arr_hydro)
        # tot_delays.append(arr_wet + arr_hydro)

    scale    = wavelength / (4 * np.pi)
    wetDelay = (wet_delays[0] - wet_delays[1]) * scale
    hydDelay = (hyd_delays[0] - hyd_delays[1]) * scale
    # totDelay = (tot_delays[0] - tot_delays[1]) * scale

    ## write to disk
    with xr.open_dataset(path_gunw, group='science/grids/data') as ds:
        lats = ds.latitude
        lons = ds.longitude

    names   = 'troposphereWet troposphereHydrostatic troposphereTotal'.split()
    lst_das = []
    for name, delays in zip(names, [wetDelay, hydDelay, totDelay]):
        da = xr.DataArray(delays, name=name, dims='latitude longitude'.split(),
                coords={'latitude': lats, 'longitude': lons})
    ds  = xr.merge(lst_ds)

    model = lst_paths[0].split('_')[0]
    dst   = os.path.join(out_dir, f'{model}_interferometric_{ref}_{sec}.nc')
    ds.to_netcdf(dst)
    logger.info ('Wrote:', dst)

    ## optionally update netcdf
    if update_flag:
        pass


if __name__ == '__main__':
    main()
