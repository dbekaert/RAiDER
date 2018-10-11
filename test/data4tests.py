# types of models
import models
import erai
import era5
import merra2
import wrf

def load_weather_model(model_type):
    if model_type=='erai' or model_type == 'era5':
        if model_type =='erai':
            model = models.ERAI()
            #model = erai.Model
            name = 'ERA-I'
        elif model_type == 'era5':
            model = models.ERA5()
            #model = era5.Model
            name = 'ERA-5'
        wm = {'type': model, 'files': None,
                'name': name}

    elif model_type =='pickle':
        wm = {'type': 'pickle', 'files': None,
              'name': 'ERA-I'}
    elif model_type =='wrf':
        wm = {'type': 'wrf', 'files': ['test/scenario_5/wrfout_d02_2010-06-24_13:16:00',
                'test/scenario_5/wrfplev_d02_2010-06-24_13:16:00'],
              'name': 'WRF'}
    else:
#        print('Attempted to use model_type {}'.format(model_type))
        raise RuntimeError('Not a valid weather model')

    return wm



    refFile = '/home/maurer/software/pythonmodules/raytracing/RESULTS_TEST/orig/ERA-I_hydro_2018-01-01T00:48:00_std.ENVI'
    testFile = '/home/maurer/software/pythonmodules/raytracing/test/ERA-I_hydro_2018-01-01T00:48:00_std.ENVI'

    #hydrods = gdal.Open(self.refFile)
    #refData = hydrods.GetRasterBand(1).ReadAsArray()
    #hydrods = gdal.Open(self.testFile)
    #testData = hydrods.GetRasterBand(1).ReadAsArray()
    #hydrods = None
        
