
import utils.demdownload as dld

def readLL(lat, lon):
    '''
    Parse lat/lon/height inputs and return 
    the appropriate outputs
    '''
    # Lats/Lons
    if lat is None:
        # They'll get set later with weather
        lats = lons = None
        latproj = lonproj = None
    else:
        try:
            # If they are files, open them
            lats, latproj = util.gdal_open(lat, returnProj = True)
            lons, lonproj = util.gdal_open(lon, returnProj = True)
        except:
            # assume that they are numbers/list/numpy array
            lats = lat
            lons = lon
            latproj = lonproj = None
            lon = lat = None

    [lats, lons] = enforceNumpyArray(lats, lons)

    return lats, lons


def getHeights(lats, lons,heights, demLoc):
    # Height
    height_type, height_info = heights
    if height_type == 'dem':
        try:
            hts = util.gdal_open(height_info)
        except RuntimeError:
            print('WARNING: File {} could not be opened. \n')
            print('Proceeding with DEM download'.format(height_info))
            hts = dld.download_dem(lats, lons, demLoc)
    elif height_type == 'lvs':
        hts = height_info
        latlist, lonlist, hgtlist = [], [], []
        for ht in hts:
           latlist.append(lats.flatten())
           lonlist.append(lons.flatten())
           hgtlist.append(np.array([ht]*length(lats.flatten())))
        lats = np.array(latlist)
        lons = np.array(lonlist)
        hts = np.array(hgtlist)
        
    if height_type == 'download':
        hts = dld.download_dem(lats, lons, demLoc)

    [lats, lons, hts] = enforceNumpyArray(lats, lons, hts)

    return lats, lons, hts


def setGeoInfo(lat, lon, latproj, lonproj):
    # TODO: implement
    # set_geo_info should be a list of functions to call on the dataset,
    # and each will do some bit of work
    set_geo_info = list()
    if lat is not None:
        def geo_info(ds):
            ds.SetMetadata({'X_DATASET': os.path.abspath(lat), 'X_BAND': '1',
                            'Y_DATASET': os.path.abspath(lon), 'Y_BAND': '1'})
        set_geo_info.append(geo_info)
    # Is it ever possible that lats and lons will actually have embedded
    # projections?
    if latproj:
        def geo_info(ds):
            ds.SetProjection(latproj)
        set_geo_info.append(geo_info)
    elif lonproj:
        def geo_info(ds):
            ds.SetProjection(lonproj)
        set_geo_info.append(geo_info)

    return set_geo_info


def enforceNumpyArray(*args):
    '''
    Enforce that a set of arguments are all numpy arrays. 
    Raise an error on failure.
    '''
    return [checkArg(a) for a in args]

def checkArg(arg):

    if arg is None:
       return None
    else:
       import numpy as np
       try:
          return np.array(arg)
       except:
          raise RuntimeError('checkArg: Cannot covert argument to numpy arrays')


