
def AllowedModels():
    '''
    return a list of the implemented model types
    '''
    allowedModels = [
      'ERA5',
      'ERA5T',
      'ERAI',
      'MERRA2',
      'WRF',
      'HRRR',
      'GMAO',
      'HDF5']

    return allowedModels


def checkIfImplemented(modelName):
    '''
    Check whether the input model name has been implemented
    '''
    allowedWMTypes = AllowedModels()
    if modelName not in allowedWMTypes:
        raise RuntimeError('Weather model {} not allowed/implemented'.format(modelName))

