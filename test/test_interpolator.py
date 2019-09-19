# Unit and other tests
import datetime
import gdal
import math
import numpy as np
import os
import pandas as pd
import unittest

import RAiDER.llreader
import RAiDER.util
import RAiDER.delay
import RAiDER.delayFcns

from RAiDER.constants import Zenith

class FcnTests(unittest.TestCase):

    x = np.linspace(0,10,100)
    y = x.copy()
    #zlevels = np.array([-1, 1, 3, 5, 10, 20])
    zlevels =  np.arange(-1, 21, 1)
    #z = zlevels + np.random.rand(len(zlevels))
    z = zlevels
    [xs, ys, zs] = np.meshgrid(x, y, z)
    def F(x, y, z):
      return np.sin(x)*np.cos(y)*(0.1*z - 5)
    values = F(*np.meshgrid(x, y, z, indexing='ij', sparse=True))

    nanindex = np.array([[3, 2, 2],
                         [0, 0, 4],
                         [3, 0, 0],
                         [2, 4, 3],
                         [1, 0, 1],
                         [3, 0, 3],
                         [2, 1, 1],
                         [0, 2, 1],
                         [2, 1, 3],
                         [3, 0, 3]])
    nanIndex = np.zeros(values.shape)
    valuesWNans = values.copy()
    for k in range(nanindex.shape[0]):
       valuesWNans[nanindex[k,0], nanindex[k,1], nanindex[k,2]] = np.nan
       nanIndex[nanindex[k,0], nanindex[k,1], nanindex[k,2]] = 1
    nanIndex = nanIndex.astype('bool')

    testPoint1 = np.array([5, 5, 5])
    testPoint2 = np.array([4.5, 0.5, 15.0])
    trueValue1 = 1.22404
    trueValue2 = 3.00252

    tv1 = np.array([ 0.        ,  1.        ,  2.        ,  3.        ,  4.        ,
        5.        ,  0.        ,  0.84147098,  0.90929743,  0.14112001,
       -0.7568025 , -0.95892427,  0.5       ,  1.5       ,  2.5       ,
        3.5       ,  4.5       ])
    truev1 = np.array([ 0.42073549,  0.87538421,  0.52520872, -0.30784124, -0.85786338])

    # test interp_along_axis
    z2 = np.tile(np.arange(100)[...,np.newaxis], (5,1,5)).swapaxes(1,2)
    zvals = 0.3*z2 - 12.75
    newz = np.tile(np.array([1.5, 9.9, 15, 23.278, 39.99, 50.1])[...,np.newaxis], (5,1,5)).swapaxes(1,2)
    corz = 0.3*newz - 12.75

    # test error messaging
    def test_interpVector(self):
        out =RAiDER.interpolator.interpVector(self.tv1, 6)
        self.assertTrue(np.allclose(out, self.truev1))
    def test_interp3D_1(self):
        f = RAiDER.interpolator._interp3D(self.x, self.y, self.z, self.values, self.zlevels)
        self.assertTrue((np.abs(f(self.testPoint1) - self.trueValue1) < 0.01)[0])
    def test_interp3D_2(self):
        f = RAiDER.interpolator._interp3D(self.x, self.y, self.z, self.values, self.zlevels)
        self.assertTrue((np.abs(f(self.testPoint2) - self.trueValue2) < 0.01)[0])
    def test_fillna3D(self):
        final = RAiDER.interpolator.fillna3D(self.valuesWNans)
        denom = np.abs(self.values[self.nanIndex])
        error = np.abs(final[self.nanIndex] - self.values[self.nanIndex])/np.where(denom==0, 1, denom)
        self.assertTrue(np.mean(error)<0.1)
    def test_interp_along_axis(self):
        out = RAiDER.interpolator.interp_along_axis(self.z2, self.newz, self.zvals, axis = 2)
        self.assertTrue(np.allclose(self.corz, out))
  


def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

