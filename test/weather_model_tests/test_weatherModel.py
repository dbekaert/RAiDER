# Unit and other tests
import datetime
import glob
import math
import os
import pickle
import unittest

import numpy as np
from osgeo import gdal
from scipy.interpolate import LinearNDInterpolator as lndi

from RAiDER.constants import Zenith
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import modelName2Module, make_weather_model_filename


class WMTests(unittest.TestCase):

    time = datetime.datetime(2018, 1, 1, 2, 0, 0)

    lat_box = np.array([16, 18])
    lon_box = np.array([-103, -100])
    time = datetime.datetime(2018, 1, 1)

    lats_shape = (11, 15)
    lons_shape = (11, 15)

    lats_hrrr = np.array([36.5, 37.5])
    lons_hrrr = np.array([-77, -76])

    basedir = os.path.join('test', 'scenario_3')
    wmLoc = os.path.join(basedir, 'weather_files')
    model_module_name, model_obj = modelName2Module('ERA5')
    era5 = {'type': model_obj(), 'files': None, 'name': 'ERA5'}


def prepareWeatherModel(weatherDict, wmFileLoc, out, lats=None, lons=None,
                        los=None, zref=None, time=None, verbose=False,
                        download_only=False, makePlots=False):

    def test_noNaNs(self):
        self.assertTrue(np.sum(np.isnan(self.weather_model._xs)) == 0)
        self.assertTrue(np.sum(np.isnan(self.weather_model._ys)) == 0)
        self.assertTrue(np.sum(np.isnan(self.weather_model._zs)) == 0)
        self.assertTrue(np.sum(np.isnan(self.weather_model._p)) == 0)
        self.assertTrue(np.sum(np.isnan(self.weather_model._e)) == 0)
        self.assertTrue(np.sum(np.isnan(self.weather_model._t)) == 0)
        self.assertTrue(
            np.sum(np.isnan(self.weather_model._wet_refractivity)) == 0)
        self.assertTrue(
            np.sum(np.isnan(self.weather_model._hydrostatic_refractivity)) == 0)


    def test_makeWMFilename(self):
        self.assertTrue(
            make_weather_model_filename('ERA5', datetime.datetime(2020, 1, 1, 0, 0, 0), (15, 17, -72, -70)) ==
            'ERA5_2020-01-01T00_00_00_15N_17N_-72E_-70E.h5'
        )

    @unittest.skip("skipping full model test until all other unit tests pass")
    def test_prepareWeatherModel_ERA5(self):
        model_module_name, model_obj = modelName2Module('ERA5')
        basedir = os.path.join('test', 'scenario_1')
        wmFileLoc = os.path.join(basedir, 'weather_files')
        #era5 = {'type': model_obj(), 'files': None, 'name': 'ERA5'}
        era5 = {'type': model_obj(), 'files': glob.glob(
            wmFileLoc + os.sep + '*.nc'), 'name': 'ERA5'}

        weather_model, lats, lons = prepareWeatherModel(
            era5, wmFileLoc, basedir, verbose=True)
        #weather_model, lats, lons = prepareWeatherModel(era5,wmFileLoc, basedir, verbose=True, lats = self.lat_box, lons = self.lon_box, time = self.time)
        self.assertTrue(lats.shape == self.lats_shape)
        self.assertTrue(lons.shape == self.lons_shape)
        self.assertTrue(lons.shape == lats.shape)
        self.assertTrue(
            weather_model._wet_refractivity.shape[:2] == self.lats_shape)
        self.assertTrue(weather_model.Model() == 'ERA-5')

    @unittest.skip("skipping full model test until all other unit tests pass")
    def test_prepareWeatherModel_HRRR(self):
        model_module_name, model_obj = modelName2Module('HRRR')
        basedir = os.path.join('test', 'scenario_2')
        wmFileLoc = os.path.join(basedir, 'weather_files')
        #hrrr = {'type': model_obj(), 'files': None, 'name': 'HRRR'}
        hrrr = {'type': model_obj(), 'files': glob.glob(
            wmFileLoc + os.sep + '*.nc'), 'name': 'HRRR'}

        weather_model, lats, lons = prepareWeatherModel(
            hrrr, wmFileLoc, basedir, verbose=True, lats=self.lats_hrrr, lons=self.lons_hrrr)
        self.assertTrue(np.all(lons.shape == lats.shape))
        self.assertTrue(weather_model.Model() == 'HRRR')


def main():
    unittest.main()


if __name__ == '__main__':

    unittest.main()
