
def AllowedModels():
    '''
    return a list of the implemented model types
    '''
    allowedModels = [
      'ERA-5',
      'ERA-I',
      'MERRA-2',
      'WRF',
      'HRRR',
      'pickle',
      'grib']

    return allowedModels


def checkIfImplemented(modelName):
    '''
    Check whether the input model name has been implemented
    '''
    allowedWMTypes = AllowedModels()
    if modelName not in allowedWMTypes:
        raise RuntimeError('Weather model {} not allowed/implemented'.format(weather_fmt))


