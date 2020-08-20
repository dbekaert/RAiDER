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
from RAiDER.utilFcns import modelName2Module


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

    @unittest.skip('skipping full model test until all other unit tests pass')
    def test_interpVector(self):
        wm = self.weather_model
        [X, Y, Z] = np.meshgrid(wm._xs, wm._ys, wm._zs)
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        wrf = wm._wet_refractivity
        hrf = wm._hydrostatic_refractivity
        zs = wm._zs
        zref = 15000
        zmask = zs < zref
        stepSize = 10

        f1 = lndi(points, wrf.flatten())
        f2 = lndi(points, hrf.flatten())
        zint = zs[zmask]
        ray = np.stack([-101 * np.ones(len(zint)), 17 *
                        np.ones(len(zint)), zint]).T
        testwet = f1(ray)
        testhydro = f2(ray)
        dx = ray[1, 2] - ray[0, 2]
        mask = np.isnan(testwet) | np.isnan(testhydro)
        totalwet = 1e-6 * dx * np.sum(testwet[~mask])
        totalhydro = 1e-6 * dx * np.sum(testhydro[~mask])
        totalwet = 1e-6 * np.trapz(testwet[~mask], zint[~mask])
        totalhydro = 1e-6 * np.trapz(testhydro[~mask], zint[~mask])

        total_wet = 1e-6 * np.trapz(wrf[5, 9, zmask], zs[zmask])
        total_hydro = 1e-6 * np.trapz(hrf[5, 9, zmask], zs[zmask])

        self.assertTrue(np.abs(totalwet - total_wet) < 0.01)
        self.assertTrue(np.abs(totalhydro - total_hydro) < 0.01)

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
