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
    zlevels = np.array([-1, 1, 3, 5, 10, 20])
    z = zlevels + np.random.rand(6)
    [xs, ys, zs] = np.meshgrid(x, y, z)
    def F(x, y, z):
      return np.sin(x)*np.cos(y)*(0.1*z - 5)
    values = F(*np.meshgrid(x, y, z, indexing='ij', sparse=True))

    testPoint1 = np.array([5, 5, 5])
    testPoint2 = np.array([4.5, 0.5, 15.0])
    trueValue1 = 1.22404
    trueValue2 = 3.00252

    tv1 = np.array([ 0.        ,  1.        ,  2.        ,  3.        ,  4.        ,
        5.        ,  0.        ,  0.84147098,  0.90929743,  0.14112001,
       -0.7568025 , -0.95892427,  0.5       ,  1.5       ,  2.5       ,
        3.5       ,  4.5       ])
    truev1 = np.array([ 0.42073549,  0.87538421,  0.52520872, -0.30784124, -0.85786338])

    # test error messaging
    def test_interpVector(self):
        out =RAiDER.interpolator.interpVector(self.tv1, 6)
        self.assertTrue(np.allclose(out, self.truev1))
    def test_interp3D_1(self):
        f = RAiDER.interpolator._interp3D(self.xs, self.ys, self.zs, self.values, self.zlevels)
        self.assertTrue((np.abs(f(self.testPoint1) - self.trueValue1) < 0.01)[0])
    def test_interp3D_2(self):
        f = RAiDER.interpolator._interp3D(self.xs, self.ys, self.zs, self.values, self.zlevels)
        self.assertTrue((np.abs(f(self.testPoint2) - self.trueValue2) < 0.01)[0])

def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

