# Unit and other tests
import datetime
import gdal
import glob
import math
import numpy as np
import os
from scipy.interpolate import LinearNDInterpolator as lndi
import pickle
import unittest

from RAiDER.utilFcns import modelName2Module, writeLL
from RAiDER.processWM import prepareWeatherModel
from RAiDER.constants import Zenith


class WMTests(unittest.TestCase):

    time = datetime.datetime(2018,1,1,2,0,0)

    lat_box = np.array([16, 18])
    lon_box = np.array([-103, -100])
    time = datetime.datetime(2018,1,1)

    lats_shape = (11,15)
    lons_shape = (11,15)

    lats_hrrr = np.array([36.5, 37.5])
    lons_hrrr = np.array([-77, -76])

    # test error messaging
    def test_interpVector(self):
        picklefile = os.path.join('test', 'scenario_0', 'pickledWeatherModel.pik')
        with open(picklefile, 'rb') as f:
            wm = pickle.load(f)
        points = np.stack([wm._xs.flatten(), wm._ys.flatten(), wm._zs.flatten()], axis = -1)
        wrf = wm._wet_refractivity                                                                            
        hrf = wm._hydrostatic_refractivity
        zs = wm._zs[1,1,:]
        zref = 15000
        stepSize = 10

        f1 = lndi(points, wrf.flatten()) 
        f2 = lndi(points, hrf.flatten())  
        ray = np.stack([-100*np.ones(zref//100), 20*np.ones(zref//100), 
                         np.linspace(-100, zref, zref//100)]).T
        testwet = f1(ray)
        testhydro = f2(ray)
        dx = ray[1,2] - ray[0,2] 
        total = 1e-6*dx*np.sum(testwet + testhydro)
        total_true = 1e-6*(np.trapz(wrf[1,1,:], zs) + np.trapz(hrf[1,1,:], zs))

        self.assertTrue(np.abs(total-total_true) < 0.01)

    def test_prepareWeatherModel_ERA5(self):
        model_module_name, model_obj = modelName2Module('ERA5')
        basedir = os.path.join('test', 'scenario_1')
        wmFileLoc = os.path.join(basedir, 'weather_files')
        #era5 = {'type': model_obj(), 'files': None, 'name': 'ERA5'}
        era5 = {'type': model_obj(), 'files': glob.glob(wmFileLoc + os.sep + '*.nc'), 'name': 'ERA5'}

        weather_model, lats, lons = prepareWeatherModel(era5,wmFileLoc, basedir, verbose=True)
        #weather_model, lats, lons = prepareWeatherModel(era5,wmFileLoc, basedir, verbose=True, lats = self.lat_box, lons = self.lon_box, time = self.time)
        import pdb; pdb.set_trace()
        self.assertTrue(lats.shape == self.lats_shape)
        self.assertTrue(lons.shape == self.lons_shape)
        self.assertTrue(lons.shape == lats.shape)
        self.assertTrue(weather_model._wet_refractivity.shape[:2] == self.lats_shape)
        self.assertTrue(weather_model.Model()=='ERA-5')

    def test_prepareWeatherModel_HRRR(self):
        model_module_name, model_obj = modelName2Module('HRRR')
        basedir = os.path.join('test', 'scenario_2')
        wmFileLoc = os.path.join(basedir, 'weather_files')
        #hrrr = {'type': model_obj(), 'files': None, 'name': 'HRRR'}
        hrrr = {'type': model_obj(), 'files': glob.glob(wmFileLoc + os.sep + '*.nc'), 'name': 'HRRR'}

        weather_model, lats, lons = prepareWeatherModel(hrrr,wmFileLoc, basedir, verbose=True)
        #weather_model, lats, lons = prepareWeatherModel(hrrr,wmFileLoc, basedir, verbose=True, lats = self.lats_hrrr, lons = self.lons_hrrr, time = self.time)
        self.assertTrue(np.all(lons.shape == lats.shape))
        self.assertTrue(weather_model.Model()=='HRRR')

def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

