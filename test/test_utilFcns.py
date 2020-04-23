# Unit and other tests
import datetime
import gdal
import math
import numpy as np
import os
import pandas as pd
import unittest


class FcnTests(unittest.TestCase):

    # test sind
    theta1 = np.array([0,30,90,180])
    truev1 = np.array([0,0.5,1,0])
    
    # test cosd
    theta2 = np.array([0,60,90,180])
    truev2 = np.array([1,0.5,0,-1])
    
    # test tand
    theta3 = np.array([0,30,45,60])
    truev3 = np.array([0,1/np.sqrt(3),1,np.sqrt(3)])
    
    # test gdal_open
    fname1 = '../test/test_geom/lat.rdr'
    shape1 = (45,226)
    
    # test writeResultsToHDF5
    lats = np.array([15.0, 15.5, 16.0, 16.5, 17.5, -40, 60, 90])
    lons = np.array([-100.0, -100.4, -91.2, 45.0, 0., -100,-100, -100])
    hgts = np.array([0., 1000., 10000., 0., 0., 0., 0., 0.])
    wet = np.zeros(lats.shape)
    hydro = np.ones(lats.shape)
    filename1 = 'dummy.hdf5'
    
    # test writeArrayToRaster
    array = np.transpose(np.array([np.arange(0,10)]))*np.arange(0,10)
    filename2 = 'dummy.out'

    # test error messaging
    def test_sind(self):
        from RAiDER.utilFcns import sind
        out =sind(self.theta1)
        self.assertTrue(np.allclose(out, self.truev1))

    def test_cosd(self):
        from RAiDER.utilFcns import cosd
        out =cosd(self.theta2)
        self.assertTrue(np.allclose(out, self.truev2))

    def test_tand(self):
        from RAiDER.utilFcns import tand
        out =tand(self.theta3)
        self.assertTrue(np.allclose(out, self.truev3))

    def test_gdal_open(self):
        from RAiDER.utilFcns import gdal_open
        out = gdal_open(self.fname1, False)
        self.assertTrue(np.allclose(out.shape, self.shape1))

    def test_writeResultsToHDF5(self):
        from RAiDER.utilFcns import writeResultsToHDF5
        writeResultsToHDF5(self.lats, self.lons, self.hgts, self.wet, self.hydro, self.filename1)
        import h5py
        with h5py.File(self.filename1, 'r') as f:
            lats = np.array(f['lat'])
            hydro = np.array(f['hydroDelay'])
            delayType = f.attrs['DelayType']
        self.assertTrue(np.allclose(lats, self.lats))
        self.assertTrue(np.allclose(hydro, self.hydro))
        self.assertEqual(delayType, 'Zenith')

    def test_writeArrayToRaster(self):
        from RAiDER.utilFcns import writeArrayToRaster
        writeArrayToRaster(self.array, self.filename2)
        with gdal.Open(self.filename2, gdal.GA_ReadOnly) as ds:
            b = ds.GetRasterBand(1)
            d = b.ReadAsArray()
            nodata = b.GetNoDataValue()
            self.assertTrue(np.allclose(d, self.array))
            self.assertEqual(nodata, 0)

  


def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

