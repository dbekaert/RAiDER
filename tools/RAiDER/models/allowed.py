
def AllowedModels():
    '''
    return a list of the implemented model types
    '''
    allowedModels = [
      'ERA5',
      'ERAI',
      'MERRA2',
      'WRF',
      'HRRR',
      'PICKLE',
      'GRIB']

    return allowedModels


def checkIfImplemented(modelName):
    '''
    Check whether the input model name has been implemented
    '''
    allowedWMTypes = AllowedModels()
    if modelName not in allowedWMTypes:
        raise RuntimeError('Weather model {} not allowed/implemented'.format(modelName))


