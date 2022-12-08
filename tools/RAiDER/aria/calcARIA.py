"""
Calculate the interferometric phase from the 4 delays files of a GUNW
Write it to disk
"""
import xarray as xr
import numpy as np

## z is a placeholder
def get_delay(delayFile):
    ds = xr.open_dataset(delayFile)
    wet   = ds['wet'].isel(z=0).data
    hydro = ds['hydro'].isel(z=0).data
    tot   = wet + hydro
    return tot


def main(delayFiles, update_nc):
    """ Pull the wet and hydrostatic delays from the netcdf

    Calculate the interferometric phase delay
    Write to disk, and optionally update the netcdf
    """
    delays = np.stack([get_delay(delayFile) for delayFile in delayFiles])
    breakpoint()





if __name__ == '__main__':
    main()
