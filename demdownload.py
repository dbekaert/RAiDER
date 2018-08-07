from osgeo import gdal
import numpy as np
import scipy.interpolate
import util


_world_dem = ('https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/'
              'SRTM_GL1_Ellip/SRTM_GL1_Ellip_srtm.vrt')


def download_dem(lats, lons):
    minlon = np.min(lons)
    maxlon = np.max(lons)
    minlat = np.min(lats)
    maxlat = np.max(lats)
    # We'll download at a high resolution to minimize the losses during
    # interpolation
    lonsteps = lons.shape[0] * 5
    latsteps = lats.shape[1] * 5
    lonres = (maxlon - minlon) / lonsteps
    latres = (maxlat - minlat) / latsteps

    gdal.Warp(
            '/vsimem/warped', f'/vsicurl/{_world_dem}',
            options=f'-te {minlon} {minlat} {maxlon} {maxlat} '
                    f'-tr {lonres} {latres}')
    try:
        out = util.gdal_open('/vsimem/warped')
    finally:
        # Have to make sure the file gets cleaned up
        gdal.Unlink('/vsimem/warped')

    # For some reason, GDAL writes the file top to bottom, so we'll flip
    # it
    out = out[::-1]

    # Index into out as lat, lon
    interpolator = scipy.interpolate.RegularGridInterpolator(
            (np.linspace(minlat, maxlat, latsteps),
                np.linspace(minlon, maxlon, lonsteps)), out, method='linear')

    out_interpolated = interpolator(np.stack((lats, lons), axis=-1))

    return out_interpolated
