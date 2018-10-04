from osgeo import gdal
import numpy as np
import scipy.interpolate
import util

gdal.UseExceptions()


_world_dem = ('https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/'
              'SRTM_GL1_Ellip/SRTM_GL1_Ellip_srtm.vrt')


def download_dem(lats, lons, save_flag= True):
    print('Getting the DEM')
    # We'll download at a high resolution to minimize the losses during
    # interpolation
    if len(lats) > 1:
        minlon = np.nanmin(lons)
        maxlon = np.nanmax(lons)
        minlat = np.nanmin(lats)
        maxlat = np.nanmax(lats)
        lonsteps = lons.shape[0] * 5
        latsteps = lats.shape[1] * 5
        lonres = (maxlon - minlon) / lonsteps
        latres = (maxlat - minlat) / latsteps
    else:
        #TODO: fix this for single inputs
        raise RuntimeError('Not yet implemented')
        minlon = np.nanmin(lons) - 0.5
        maxlon = np.nanmax(lons) + 0.5
        minlat = np.nanmin(lats) - 0.5
        maxlat = np.nanmax(lats) + 0.5
        lonres = 0.01
        latres = 0.01
        lonsteps =(maxlon - minlon) / lonres
        latsteps = (maxlat - minlat) / latres
        lats = np.linspace(minlat, maxlat, latsteps)
        lons = np.linspace(minlon, maxlon, lonsteps)

    outRaster = '/vsimem/warpedDEM'

    inRaster ='/vsicurl/{}'.format(_world_dem) 

    print('Beginning DEM download and warping')
    wrpOpt = gdal.WarpOptions(outputBounds = (minlon, minlat,maxlon, maxlat), 
                              xRes = lonres, yRes = latres)
    gdal.Warp(outRaster, inRaster, options = wrpOpt)
#    gdal.Warp(inRaster, outRaster, 
#            options='-te {minlon} {minlat} {maxlon} {maxlat} '.format(minlat = minlat, minlon =minlon, maxlon = maxlon, maxlat = maxlat) +\
#                    '-tr {lonres} {latres}'.format(lonres = lonres, latres = latres))
    print('DEM download finished')

    try:
        out = util.gdal_open(outRaster)
    except:
        raise RuntimeError('demdownload: Cannot open the warped file')
    finally:
        # Have to make sure the file gets cleaned up
        try:
            gdal.Unlink('/vsimem/warpedDEM')
        except:
            pass

    # For some reason, GDAL writes the file top to bottom, so we'll flip
    # it
    out = out[::-1]

    # Index into out as lat, lon
    print('Beginning interpolation')

    interpolator = scipy.interpolate.RegularGridInterpolator(
            points = (np.linspace(minlat, maxlat, latsteps),
                      np.linspace(minlon, maxlon, lonsteps)), 
            values = out,
            method='linear', 
            bounds_error = False)

    out_interpolated = interpolator(np.stack((lats, lons), axis=-1))

    print('Interpolation finished')

    if save_flag:
        print('Saving DEM to disk')
        outRasterName = 'warpedDEM.dem'
        util.writeArrayToRaster(out_interpolated, outRasterName)
        print('Finished saving DEM to disk')

    return out_interpolated
